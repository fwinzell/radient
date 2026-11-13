# datasets/foundation_feature_dataset.py

from typing import Tuple, Optional

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

from .generic import GenericRadiologyDataset

import time
from contextlib import contextmanager

from scipy.ndimage import label


class FeatureExtractionDataset(Dataset):
    """
    Modality-neutral spatial preprocessing for 3D feature extraction.

    Strategy
    --------
    If segmentation exists:
        configured segmentation label -> largest axial area -> ROI centroid
        -> fixed physical crop

    If segmentation does not exist:
        fallback_mode='center_crop':
            fixed physical crop centered on the volume

        fallback_mode='whole_volume':
            entire volume is resampled to target shape

    Output:
        torch.Tensor of shape [1, D, H, W]
    """

    def __init__(
        self,
        file_obj,
        orientation: str = "RAS",

        # Physical crop size in XYZ
        crop_size_mm: Tuple[float, float, float] = (
            140.0,
            140.0,
            160.0,
        ),

        # Final spatial size in ZYX 
        target_size: Tuple[int, int, int] = (
            32,
            128,
            128
        ),

        # Padding value used outside the image during cropping/resampling.
        pad_value: float = -1000.0,

        # HU preprocessing
        window_level: Optional[float] = 50.0,
        window_width: Optional[float] = 500.0,

        # Intensity policy: CT windowing, no change, per-volume z-score or min-max.
        intensity_mode: str = "ct_window",

        # Segmentation label
        lesion_label: int = 2,

        # What to do without segmentation
        fallback_mode: str = "center_crop",

        # Whether to return metadata
        return_metadata: bool = True,

        **kwargs,
    ):
        super().__init__()

        if fallback_mode not in {
            "center_crop",
            "whole_volume",
        }:
            raise ValueError(
                "fallback_mode must be 'center_crop' or 'whole_volume'"
            )

        if intensity_mode not in {"ct_window", "none", "zscore", "minmax"}:
            raise ValueError("intensity_mode must be 'ct_window', 'none', 'zscore', or 'minmax'")

        print("Initializing base dataset...")
        self.base_dataset = GenericRadiologyDataset(
            file_obj=file_obj,
            orientation=orientation,
            **kwargs,
        )
        print("Finished")

        self.crop_size_mm = crop_size_mm
        self.target_size = target_size

        self.pad_value = pad_value

        self.window_level = window_level
        self.window_width = window_width
        self.intensity_mode = intensity_mode

        self.lesion_label = lesion_label

        self.fallback_mode = fallback_mode
        self.return_metadata = return_metadata
        print(f"FeatureExtractionDataset initialized with {self.__len__()} instances")

    def __len__(self):
        return len(self.base_dataset)

    # ==============================================================
    # Centers
    # ==============================================================

    @staticmethod
    def _volume_center(
        shape_zyx,
    ):
        """
        Geometric center of the CT volume.

        Returns (x, y, z).
        """
        z, y, x = shape_zyx

        return (
            x // 2,
            y // 2,
            z // 2,
        )

    def _lesion_center(
        self,
        seg: np.ndarray,
    ):
        """
        Find the center of the configured segmentation label.

        1. Find axial slice with the largest lesion area.
        2. Calculate centroid within that slice.
        """
        mask = seg == self.lesion_label

        if not np.any(mask):
            return None

        slice_area = mask.sum(axis=(1, 2))
        z = int(np.argmax(slice_area))

        ys, xs = np.where(mask[z])

        if len(xs) == 0:
            return None

        x = int(np.round(xs.mean()))
        y = int(np.round(ys.mean()))

        return x, y, z

    # ==============================================================
    # Physical crop
    # ==============================================================

    def _crop_physical(
        self,
        ct: np.ndarray,
        center_xyz,
        spacing_xyz,
    ):
        """
        Crop a physical [X,Y,Z] box around center_xyz.

        ct:
            [Z,Y,X]

        center_xyz:
            (X,Y,Z)

        spacing_xyz:
            (SX,SY,SZ)
        """

        x, y, z = center_xyz
        sx, sy, sz = spacing_xyz

        size_x_mm, size_y_mm, size_z_mm = self.crop_size_mm

        # Number of voxels corresponding to physical crop size.
        nx = int(np.round(size_x_mm / sx))
        ny = int(np.round(size_y_mm / sy))
        nz = int(np.round(size_z_mm / sz))

        # Ensure even dimensions.
        nx += nx % 2
        ny += ny % 2
        nz += nz % 2

        hx = nx // 2
        hy = ny // 2
        hz = nz // 2

        # Desired crop bounds
        x0 = x - hx
        x1 = x + hx

        y0 = y - hy
        y1 = y + hy

        z0 = z - hz
        z1 = z + hz

        # Amount of padding
        px0 = max(0, -x0)
        px1 = max(0, x1 - ct.shape[2])

        py0 = max(0, -y0)
        py1 = max(0, y1 - ct.shape[1])

        pz0 = max(0, -z0)
        pz1 = max(0, z1 - ct.shape[0])

        # Clamp to valid image region
        cx0 = max(0, x0)
        cx1 = min(ct.shape[2], x1)

        cy0 = max(0, y0)
        cy1 = min(ct.shape[1], y1)

        cz0 = max(0, z0)
        cz1 = min(ct.shape[0], z1)

        crop = ct[
            cz0:cz1,
            cy0:cy1,
            cx0:cx1,
        ]

        crop = np.pad(
            crop,
            (
                (pz0, pz1),
                (py0, py1),
                (px0, px1),
            ),
            mode="constant",
            constant_values=self.pad_value,
        )

        return crop

    # ==============================================================
    # Whole-volume fallback
    # ==============================================================

    def _resample_array(
        self,
        volume: np.ndarray,
        interpolation=sitk.sitkLinear,
    ):
        """
        Resample a [D,H,W] array to target [D,H,W].
        """

        target_d, target_h, target_w = self.target_size

        img = sitk.GetImageFromArray(
            volume.astype(np.float32)
        )

        original_size = img.GetSize()
        original_spacing = img.GetSpacing()

        target_size_xyz = (
            target_w,
            target_h,
            target_d,
        )

        # Preserve physical extent approximately.
        target_spacing = (
            original_spacing[0]
            * original_size[0]
            / target_size_xyz[0],

            original_spacing[1]
            * original_size[1]
            / target_size_xyz[1],

            original_spacing[2]
            * original_size[2]
            / target_size_xyz[2],
        )

        resampler = sitk.ResampleImageFilter()

        resampler.SetSize(target_size_xyz)
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetInterpolator(interpolation)
        resampler.SetDefaultPixelValue(self.pad_value)

        output = resampler.Execute(img)

        return sitk.GetArrayFromImage(output).astype(np.float32)

    # ==============================================================
    # Crop/resample to final size
    # ==============================================================

    def _resize_crop(
        self,
        crop: np.ndarray,
    ):
        """
        Convert crop to the exact final [D,W,H].
        """

        return self._resample_array(
            crop,
            interpolation=sitk.sitkLinear
        )

    # ==============================================================
    # HU normalization
    # ==============================================================

    def _normalize_intensity(
        self,
        image: np.ndarray,
    ):
        """
        Apply the selected intensity transform. CT windowing remains the
        default for backward compatibility; MRI/PET users should select a
        policy appropriate to their data explicitly.

        Example for kidney soft tissue:
        WL=50, WW=500 gives:
            lower = -200 HU
            upper = +300 HU

        Mapping -> [0,1].
        """

        image = image.astype(np.float32, copy=False)
        if self.intensity_mode == "none":
            return image
        if self.intensity_mode == "zscore":
            std = float(image.std())
            return (image - float(image.mean())) / max(std, 1e-8)
        if self.intensity_mode == "minmax":
            lo, hi = float(image.min()), float(image.max())
            if hi <= lo:
                return np.zeros_like(image)
            return (image - lo) / (hi - lo)

        if (
            self.window_level is None
            or self.window_width is None
        ):
            return image

        lower = (
            self.window_level
            - self.window_width / 2.0
        )

        upper = (
            self.window_level
            + self.window_width / 2.0
        )

        ct = (image - lower) / (upper - lower)

        ct = np.clip(
            ct,
            0.0,
            1.0,
        )

        return ct.astype(np.float32)

    # Timer function
    @staticmethod
    @contextmanager
    def timer(label="Block"):
        print(label)
        start = time.perf_counter()
        yield
        end = time.perf_counter()
        print("\r" + " " * len(label) + "\r", end="")
        print(f"{label}: {end - start:.4f}s")

    # ==============================================================
    # Final formatting
    # ==============================================================

    @staticmethod
    def _to_tensor(
        volume_dhw: np.ndarray,
    ) -> torch.Tensor:
        """
        [D,H,W] -> [C,D,H,W]
        """

        volume = volume_dhw[None, ...]

        return torch.from_numpy(
            np.ascontiguousarray(volume)
        ).float()

    # ==============================================================
    # Main dataset
    # ==============================================================

    def __getitem__(self, index):

        
        sample = self.base_dataset[index]

        pid = sample["pid"]
        sid = sample["sid"]
        barcode = sample["barcode"]

        ct = sample["ct"]
        seg = sample["seg"]
        base_meta = sample["metadata"]

        spacing = base_meta["spacing"]

        # ----------------------------------------------------------
        # 1. Determine center
        # ----------------------------------------------------------

        
        center_source = "volume_center"

        if seg is not None:

            center = self._lesion_center(
                seg
            )

            if center is not None:
                center_source = "lesion_class_2"
            else:
                center = self._volume_center(
                    ct.shape
                )

        else:

            center = self._volume_center(
                ct.shape
            )

        # ----------------------------------------------------------
        # 2. Spatial preprocessing
        # ----------------------------------------------------------

        if (
            seg is None
            and self.fallback_mode == "whole_volume"
        ):
            # Use entire available CT volume.
            processed = self._resample_array(
                ct,
                interpolation=sitk.sitkLinear,
            )

            spatial_mode = "whole_volume"

        else:

            # Fixed physical crop around lesion or volume center.
            cropped = self._crop_physical(
                ct=ct,
                center_xyz=center,
                spacing_xyz=spacing,
            )

            processed = self._resize_crop(
                cropped
            )

            spatial_mode = (
                "lesion_crop"
                if center_source == "lesion_class_2"
                else "center_crop"
            )

        # ----------------------------------------------------------
        # 3. Intensity normalization
        # ----------------------------------------------------------

        processed = self._normalize_intensity(
            processed
        )

        # ----------------------------------------------------------
        # 4. Convert [D,H,W] -> [1,D,H,W]
        # ----------------------------------------------------------

        tensor = self._to_tensor(
            processed
        )

        if not self.return_metadata:
            return {
                "pid": pid,
                "sid": sid,
                "barcode": barcode,
                "image": tensor,
            }

        metadata = {
            **base_meta,

            "center_xyz": center,
            "center_source": center_source,
            "intensity_mode": self.intensity_mode,

            "spatial_mode": spatial_mode,

            "output_shape": tuple(
                tensor.shape
            ),

            "normalized_min": float(
                processed.min()
            ),

            "normalized_max": float(
                processed.max()
            ),
        }

        return {
            "pid": pid,
            "sid": sid,
            "barcode": barcode,
            "image": tensor,
            "metadata": metadata,
        }


class KidneyFeatureDataset(Dataset):
    """
    Kidney-level preprocessing dataset for foundation-model feature extraction.

    Each case returns two kidney crops in fixed anatomical order:

        left_image
        right_image

    Each crop has shape:
        [1, D, H, W]

    where by default:
        D = 32
        H = 128
        W = 128

    If only one kidney is present, the missing slot is None.

    Segmentation labels:
        0 = background
        1 = kidney
        2 = kidney abnormality / lesion

    Kidney detection:
        labels 1 and 2 are combined to identify the anatomical kidney.
        This prevents a lesion from splitting the kidney mask.

    Crop center:
        centroid of the complete kidney component.

    The segmentation itself is NOT provided to the foundation model.
    It is used only to determine the crop.
    """

    def __init__(
        self,
        file_obj,
        orientation: str = "RAS",

        # Fixed physical field of view around each kidney
        # [W,H,D]
        crop_size_mm: Tuple[float, float, float] = (
            140.0,
            140.0,
            160.0,
        ),

        # Final model input
        # [Z,Y,X]
        target_size: Tuple[int, int, int] = (
            32,
            128,
            128
        ),

        # CT padding outside scan
        pad_value: float = -1000.0,

        # HU normalization
        window_level: Optional[float] = 50.0,
        window_width: Optional[float] = 500.0,

        # Segmentation labels
        kidney_label: int = 1,
        lesion_label: int = 2,

        # Minimum kidney component size in voxels
        min_kidney_voxels: int = 1000,

        return_metadata: bool = True,

        **kwargs,
    ):
        super().__init__()

        self.base_dataset = GenericRadiologyDataset(
            file_obj=file_obj,
            orientation=orientation,
            **kwargs,
        )
        self.orientation=orientation

        self.crop_size_mm = tuple(float(x) for x in crop_size_mm)

        self.target_size = tuple(int(x) for x in target_size)

        self.pad_value = float(pad_value)

        self.window_level = window_level
        self.window_width = window_width

        self.kidney_label = int(kidney_label)

        self.lesion_label = int(lesion_label)

        self.min_kidney_voxels = int(min_kidney_voxels)

        self.return_metadata = return_metadata

    def __len__(self):
        return len(self.base_dataset)

    # ==============================================================
    # Kidney detection
    # ==============================================================

    def _find_kidneys(
        self,
        seg: np.ndarray,
    ):
        """
        Identify individual kidneys using labels 1 + 2.

        Returns a list of dictionaries containing:

            component
            center_xyz
            centroid_x
            centroid_y
            centroid_z
            volume
            lesion_volume

        Components are later assigned left/right based on their
        x-coordinate in RAS orientation.
        """

        kidney_mask = (
            (seg == self.kidney_label)
            | (seg == self.lesion_label)
        )

        if not np.any(kidney_mask):
            return []

        # 26-connectivity is appropriate for anatomical structures.
        structure = np.ones(
            (3, 3, 3),
            dtype=np.uint8,
        )

        labeled, n_components = label(
            kidney_mask,
            structure=structure,
        )

        kidneys = []

        lesion_mask = (
            seg == self.lesion_label
        )

        for component_id in range(
            1,
            n_components + 1,
        ):

            component_mask = (
                labeled == component_id
            )

            volume = int(
                component_mask.sum()
            )

            # Ignore tiny disconnected segmentation fragments.
            if volume < self.min_kidney_voxels:
                continue

            coords = np.argwhere(
                component_mask
            )

            # coords are [z, y, x]
            centroid_zyx = coords.mean(
                axis=0
            )

            z, y, x = np.round(
                centroid_zyx
            ).astype(int)

            lesion_volume = int(
                np.count_nonzero(
                    lesion_mask
                    & component_mask
                )
            )

            kidneys.append(
                {
                    "component": component_id,
                    "center_xyz": (
                        int(x),
                        int(y),
                        int(z),
                    ),
                    "centroid_x": float(x),
                    "centroid_y": float(y),
                    "centroid_z": float(z),
                    "volume": volume,
                    "lesion_volume": lesion_volume,
                }
            )

        # Sort by x-coordinate.
        #
        # In RAS:
        #     smaller x -> right
        #     larger x  -> left
        kidneys.sort(
            key=lambda k: k["centroid_x"]
        )

        return kidneys

    def _assign_left_right(
        self,
        kidneys,
    ):
        """
        Convert kidney components into:

            left_kidney
            right_kidney

        Requires two found kidneys

        For a normal two-kidney scan, the lower-x component is
        right and the higher-x component is left in RAS coordinates.

        Returns:
            right_kidney, left_kidney
        """

        # If more than two components survived filtering, retain
        # the two largest anatomical structures.
        if len(kidneys) > 2:
            kidneys = sorted(
                kidneys,
                key=lambda k: k["volume"],
                reverse=True,
            )[:2]

        kidneys.sort(key=lambda k: k["centroid_x"])

        if self.orientation[0] == "R":
            right_kidney = kidneys[0]
            left_kidney = kidneys[1]
        elif self.orientation[0] == "L":
            right_kidney = kidneys[1]
            left_kidney = kidneys[0]


        return right_kidney, left_kidney

    
    # ==============================================================
    # Physical crop
    # ==============================================================

    def _crop_physical(
        self,
        ct: np.ndarray,
        center_xyz,
        spacing_xyz,
    ):
        """
        Crop a fixed physical region around one kidney.

        ct:
            [Z, Y, X]

        center_xyz:
            (X, Y, Z)

        spacing_xyz:
            (SX, SY, SZ)
        """

        x, y, z = center_xyz

        sx, sy, sz = spacing_xyz

        size_x_mm, size_y_mm, size_z_mm = (
            self.crop_size_mm
        )

        nx = int(
            np.round(
                size_x_mm / sx
            )
        )

        ny = int(
            np.round(
                size_y_mm / sy
            )
        )

        nz = int(
            np.round(
                size_z_mm / sz
            )
        )

        # Force even dimensions.
        nx += nx % 2
        ny += ny % 2
        nz += nz % 2

        hx = nx // 2
        hy = ny // 2
        hz = nz // 2

        x0 = x - hx
        x1 = x + hx

        y0 = y - hy
        y1 = y + hy

        z0 = z - hz
        z1 = z + hz

        # Padding required outside image.
        px0 = max(0, -x0)
        px1 = max(
            0,
            x1 - ct.shape[2],
        )

        py0 = max(0, -y0)
        py1 = max(
            0,
            y1 - ct.shape[1],
        )

        pz0 = max(0, -z0)
        pz1 = max(
            0,
            z1 - ct.shape[0],
        )

        # Clamp to image.
        cx0 = max(0, x0)
        cx1 = min(
            ct.shape[2],
            x1,
        )

        cy0 = max(0, y0)
        cy1 = min(
            ct.shape[1],
            y1,
        )

        cz0 = max(0, z0)
        cz1 = min(
            ct.shape[0],
            z1,
        )

        crop = ct[
            cz0:cz1,
            cy0:cy1,
            cx0:cx1,
        ]

        crop = np.pad(
            crop,
            (
                (pz0, pz1),
                (py0, py1),
                (px0, px1),
            ),
            mode="constant",
            constant_values=self.pad_value,
        )

        return crop

    # ==============================================================
    # Resampling
    # ==============================================================

    def _resample_array(
        self,
        volume: np.ndarray
    ):
        """
        Resize [D,H,W] to target [D,H,W].
        """

        target_d, target_h, target_w = self.target_size

        image = sitk.GetImageFromArray(volume.astype(np.float32))

        original_size = image.GetSize()
        original_spacing = image.GetSpacing()

        target_size_xyz = (
            target_w,
            target_h,
            target_d,
        )

        target_spacing = (
            original_spacing[0] * original_size[0] / target_size_xyz[0],

            original_spacing[1] * original_size[1] / target_size_xyz[1],

            original_spacing[2] * original_size[2] / target_size_xyz[2],
        )

        resampler = sitk.ResampleImageFilter()

        resampler.SetSize(target_size_xyz)
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(self.pad_value)

        output = resampler.Execute(image)

        return sitk.GetArrayFromImage(output).astype(np.float32)

    # ==============================================================
    # HU normalization
    # ==============================================================

    def _normalize_hu(
        self,
        volume,
    ):
        """
        Map WL/WW window to [0,1].

        Default:
            WL = 50
            WW = 500

            [-200, 300] HU -> [0, 1]
        """

        if (self.window_level is None or self.window_width is None):
            return volume.astype(np.float32)

        lower = self.window_level - self.window_width / 2.0

        upper = self.window_level + self.window_width / 2.0
        
        volume = (volume - lower) / (upper - lower)

        volume = np.clip(
            volume,
            0.0,
            1.0,
        )

        return volume.astype(
            np.float32
        )

    # ==============================================================
    # Tensor conversion
    # ==============================================================

    @staticmethod
    def _to_tensor(
        volume,
    ):
        """
        [D,H,W] -> [1,D,H,W]
        """
        volume = np.ascontiguousarray(volume[None, ...])
        return torch.from_numpy(volume).float()

    # ==============================================================
    # One kidney -> one processed crop
    # ==============================================================

    def _process_kidney(
        self,
        ct,
        center_xyz,
        spacing_xyz,
    ):
        crop = self._crop_physical(
            ct=ct,
            center_xyz=center_xyz,
            spacing_xyz=spacing_xyz,
        )

        crop = self._resample_array(crop)
        crop = self._normalize_hu(crop)
        crop = self._to_tensor(crop)

        return crop

    # ==============================================================
    # Main dataset
    # ==============================================================

    def __getitem__(self, index):

        sample = self.base_dataset[
            index
        ]

        pid = sample["pid"]
        sid = sample["sid"]
        barcode = sample["barcode"]

        ct = sample["ct"]
        seg = sample["seg"]

        base_metadata = sample["metadata"]

        spacing = base_metadata["spacing"]

        # ----------------------------------------------------------
        # No segmentation
        # ----------------------------------------------------------

        if seg is None:

            result = {
                "pid": pid,
                "sid": sid,
                "barcode": barcode,

                "left": None,
                "right": None,

                "left_present": False,
                "right_present": False,

                "metadata": {
                    **base_metadata,
                    "segmentation_available": False,
                    "message": (
                        "No segmentation available; no kidney crops generated."
                    ),
                },
            }

            if not self.return_metadata:
                result.pop("metadata")

            return result

        # ----------------------------------------------------------
        # Find individual kidneys
        # ----------------------------------------------------------

        kidneys = self._find_kidneys(
            seg
        )

        if len(kidneys) == 0:

            result = {
                "pid": pid,
                "sid": sid,
                "barcode": barcode,

                "left": None,
                "right": None,

                "left_present": False,
                "right_present": False,

                "metadata": {
                    **base_metadata,
                    "segmentation_available": True,
                    "n_kidneys": 0,
                },
            }

            if not self.return_metadata:
                result.pop("metadata")

            return result

        # ----------------------------------------------------------
        # Assign anatomical sides
        # ----------------------------------------------------------

        if len(kidneys) == 1:

            kidney = kidneys[0]

            image_width = ct.shape[2]
            image_center_x = (
                image_width - 1
            ) / 2.0

            # In RAS, larger x = patient's left.
            if kidney["centroid_x"] >= image_center_x:
                left_kidney = kidney
                right_kidney = None
            else:
                left_kidney = None
                right_kidney = kidney

        else:

            right_kidney, left_kidney = (
                self._assign_left_right(
                    kidneys
                )
            )

        # ----------------------------------------------------------
        # Process crops
        # ----------------------------------------------------------

        left_crop = None
        right_crop = None

        if left_kidney is not None:

            left_crop = self._process_kidney(
                ct=ct,
                center_xyz=left_kidney[
                    "center_xyz"
                ],
                spacing_xyz=spacing,
            )

        if right_kidney is not None:

            right_crop = self._process_kidney(
                ct=ct,
                center_xyz=right_kidney[
                    "center_xyz"
                ],
                spacing_xyz=spacing,
            )

        # ----------------------------------------------------------
        # Metadata
        # ----------------------------------------------------------

        metadata = {
            **base_metadata,
            "segmentation_available": True,
            "n_kidneys_detected": len(kidneys),
            "left_present": (left_kidney is not None),
            "right_present": (right_kidney is not None),

            "left_center_xyz": (
                None
                if left_kidney is None
                else left_kidney["center_xyz"]
            ),

            "right_center_xyz": (
                None
                if right_kidney is None
                else right_kidney["center_xyz"]
            ),

            "left_volume_voxels": (
                None
                if left_kidney is None
                else left_kidney["volume"]
            ),

            "right_volume_voxels": (
                None
                if right_kidney is None
                else right_kidney["volume"]
            ),

            "left_lesion_voxels": (
                None
                if left_kidney is None
                else left_kidney["lesion_volume"]
            ),

            "right_lesion_voxels": (
                None
                if right_kidney is None
                else right_kidney["lesion_volume"]
            ),
        }

        result = {
            "pid": pid,
            "sid": sid,
            "barcode": barcode,

            "left": left_crop,
            "right": right_crop,

            "left_present": (
                left_crop is not None
            ),

            "right_present": (
                right_crop is not None
            ),

            "metadata": metadata,
        }

        if not self.return_metadata:
            result.pop("metadata")

        return result


def kidney_collate_fn(batch):
    return {
        "pid": [x["pid"] for x in batch],
        "sid": [x["sid"] for x in batch],
        "barcode": [x["barcode"] for x in batch],
        "left": [x["left"] for x in batch],
        "right": [x["right"] for x in batch],
        "left_present": torch.tensor([x["left_present"] for x in batch], dtype=torch.bool),
        "right_present": torch.tensor([x["right_present"] for x in batch], dtype=torch.bool),
        "metadata": [x.get("metadata") for x in batch],
    }
