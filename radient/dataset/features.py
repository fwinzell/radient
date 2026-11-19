# datasets/foundation_feature_dataset.py

import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

from .generic import GenericRadiologyDataset
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
        pad_value: Optional[float] = None,
        modality: str = "ct",
        preprocessor=None,
        preprocessor_strategy: str = "volume",

        # Segmentation label
        lesion_label: int = 2,

        # What to do without segmentation
        fallback_mode: str = "center_crop",
        segmentation_mode: str = "external",
        totalsegmentator_rois=None,
        totalsegmentator_task: str = "total",
        totalsegmentator_fast: bool = False,

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
        if segmentation_mode not in {"none", "external", "totalsegmentator"}:
            raise ValueError("segmentation_mode must be 'none', 'external', or 'totalsegmentator'")
        modality = str(modality).lower()
        if modality not in {"ct", "mri"}:
            raise ValueError("modality must be 'ct' or 'mri'")
        if preprocessor_strategy not in {"axial", "volume"}:
            raise ValueError("preprocessor_strategy must be 'axial' or 'volume'")

        print("Initializing base dataset...")
        # Ignore intensity options from older configs; intensity transforms
        # belong in the model-specific adapter.
        kwargs.pop("window_level", None)
        kwargs.pop("window_width", None)
        kwargs.pop("intensity_mode", None)
        self.base_dataset = GenericRadiologyDataset(
            file_obj=file_obj,
            orientation=orientation,
            **kwargs,
        )
        print("Finished")

        self.crop_size_mm = crop_size_mm
        self.target_size = target_size

        self.modality = modality
        self.pad_value = float((-1000.0 if modality == "ct" else 0.0) if pad_value is None else pad_value)
        self.preprocessor = preprocessor
        self.preprocessor_strategy = preprocessor_strategy

        self.lesion_label = lesion_label

        self.fallback_mode = fallback_mode
        self.segmentation_mode = segmentation_mode
        self.totalsegmentator_rois = list(totalsegmentator_rois or ["kidney_left", "kidney_right"])
        self.totalsegmentator_task = totalsegmentator_task
        self.totalsegmentator_fast = bool(totalsegmentator_fast)
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
        Geometric center of the image volume.

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

    def _totalsegmentator_mask(self, image, metadata):
        """Run TotalSegmentator for selected ROIs and return their union on the input image grid."""
        try:
            from totalsegmentator.python_api import totalsegmentator
        except ImportError as exc:
            raise ImportError(
                "segmentation_mode='totalsegmentator' requires TotalSegmentator. "
                "Install it with `pip install TotalSegmentator`."
            ) from exc

        sitk_image = sitk.GetImageFromArray(image.astype(np.float32, copy=False))
        sitk_image.SetSpacing(tuple(metadata["spacing"]))
        sitk_image.SetOrigin(tuple(metadata["origin"]))
        sitk_image.SetDirection(tuple(metadata["direction"]))
        try:
            import torch
            device = "gpu" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        with tempfile.TemporaryDirectory(prefix="radient_totalseg_") as temp_dir:
            temp_dir = Path(temp_dir)
            input_path = temp_dir / "volume.nii.gz"
            output_dir = temp_dir / "segmentations"
            sitk.WriteImage(sitk_image, str(input_path))
            totalsegmentator(
                str(input_path), str(output_dir), task=self.totalsegmentator_task,
                fast=self.totalsegmentator_fast, roi_subset=self.totalsegmentator_rois,
                device=device,
            )
            union = np.zeros(sitk.GetArrayFromImage(sitk_image).shape, dtype=bool)
            for roi in self.totalsegmentator_rois:
                roi_path = output_dir / f"{roi}.nii.gz"
                if not roi_path.exists():
                    raise FileNotFoundError(
                        f"TotalSegmentator did not produce requested ROI '{roi}'"
                    )
                roi_image = sitk.ReadImage(str(roi_path))
                aligned = sitk.Resample(
                    roi_image, sitk_image, sitk.Transform(), sitk.sitkNearestNeighbor,
                    0, sitk.sitkUInt8,
                )
                union |= sitk.GetArrayFromImage(aligned) > 0
        return union.astype(np.int16) * int(self.lesion_label)

    # ==============================================================
    # Physical crop
    # ==============================================================

    def _crop_physical(
        self,
        image: np.ndarray,
        center_xyz,
        spacing_xyz,
    ):
        """
        Crop a physical [X,Y,Z] box around center_xyz.

        image:
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
        px1 = max(0, x1 - image.shape[2])

        py0 = max(0, -y0)
        py1 = max(0, y1 - image.shape[1])

        pz0 = max(0, -z0)
        pz1 = max(0, z1 - image.shape[0])

        # Clamp to valid image region
        cx0 = max(0, x0)
        cx1 = min(image.shape[2], x1)

        cy0 = max(0, y0)
        cy1 = min(image.shape[1], y1)

        cz0 = max(0, z0)
        cz1 = min(image.shape[0], z1)

        crop = image[
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

        image = sample["image"][0]
        seg = sample["seg"]
        base_meta = sample["metadata"]

        full_volume_preprocess = getattr(self.preprocessor, "preprocess_dataset", None)
        if full_volume_preprocess is not None:
            result = full_volume_preprocess(image, base_meta, self.modality)
            metadata_updates = {}
            if isinstance(result, dict):
                tensor = result["image"]
                metadata_updates = result.get("metadata", {})
            else:
                tensor = result
            tensor = torch.as_tensor(tensor, dtype=torch.float32)
            if tensor.ndim == 4 and tensor.shape[0] == 1:
                pass
            elif tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            else:
                raise ValueError(
                    "Dataset preprocessors must return [D,H,W] or [1,D,H,W], "
                    f"got {tuple(tensor.shape)}"
                )
            metadata = {
                **base_meta,
                **metadata_updates,
                "preprocessor": type(self.preprocessor).__name__,
                "modality": self.modality,
                "output_shape": tuple(tensor.shape),
                "processor_input_shape": tuple(tensor.shape),
                "intensity_min": float(tensor.min()),
                "intensity_max": float(tensor.max()),
                "intensity_mean": float(tensor.mean()),
                "intensity_std": float(tensor.std(unbiased=False)),
            }
            return {
                "pid": pid,
                "sid": sid,
                "barcode": barcode,
                "image": tensor,
                "metadata": metadata,
            } if self.return_metadata else {
                "pid": pid,
                "sid": sid,
                "barcode": barcode,
                "image": tensor,
            }

        if self.segmentation_mode == "totalsegmentator":
            seg = self._totalsegmentator_mask(image, base_meta)

        spacing = base_meta["spacing"]

        # ----------------------------------------------------------
        # 1. Determine center
        # ----------------------------------------------------------

        center_source = "volume_center"

        if self.segmentation_mode == "none":
            center = self._volume_center(image.shape)
            center_source = "volume_center"
        elif seg is not None:

            center = self._lesion_center(
                seg
            )

            if center is not None:
                center_source = "lesion_class_2"
            else:
                center = self._volume_center(
                    image.shape
                )

        else:

            center = self._volume_center(
                image.shape
            )

        # ----------------------------------------------------------
        # 2. Spatial preprocessing
        # ----------------------------------------------------------

        if (
            seg is None
            and self.fallback_mode == "whole_volume"
        ):
            # Use the entire available image volume.
            processed = self._resample_array(
                image,
                interpolation=sitk.sitkLinear,
            )

            spatial_mode = "whole_volume"

        else:

            # Fixed physical crop around lesion or volume center.
            cropped = self._crop_physical(
                image=image,
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
        # 3. Convert [D,H,W] -> [1,D,H,W]
        # ----------------------------------------------------------

        if self.preprocessor is None:
            if self.modality == "ct":
                processed = np.clip((processed + 1000.0) / 2000.0, 0.0, 1.0)
                intensity_transform = "clip_scale_-1000_1000_to_0_1"
                intensity_scale = "0_1"
            else:
                padding_mask = processed == self.pad_value
                valid = processed[~padding_mask]
                mean = float(valid.mean()) if valid.size else 0.0
                std = float(valid.std()) if valid.size else 1.0
                processed = (processed - mean) / max(std, 1e-8)
                processed[padding_mask] = 0.0
                intensity_transform = "per_volume_zscore"
                intensity_scale = "zscore"
        else:
            intensity_transform = "delegated_to_model_preprocessor"
            intensity_scale = "model_specific"

        processor_input_shape = tuple(processed.shape)
        tensor = self._to_tensor(processed)
        if self.preprocessor is not None:
            tensor = self._apply_model_preprocessor(tensor)

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
            "spatial_mode": spatial_mode,

            "output_shape": tuple(
                tensor.shape
            ),
            "processor_input_shape": processor_input_shape,

            "intensity_transform": intensity_transform,
            "intensity_scale": intensity_scale,
            "modality": self.modality,
            "intensity_min": float(processed.min()),
            "intensity_max": float(processed.max()),
            "intensity_mean": float(processed.mean()),
            "intensity_std": float(processed.std()),
            "pad_value": float(self.pad_value),
        }

        return {
            "pid": pid,
            "sid": sid,
            "barcode": barcode,
            "image": tensor,
            "metadata": metadata,
        }

    def _apply_model_preprocessor(self, tensor):
        """Apply an adapter processor to a whole volume or axial slices."""
        if self.preprocessor_strategy == "axial":
            processed_slices = []
            for image_slice in tensor.permute(1, 0, 2, 3):
                result = self.preprocessor(image_slice.squeeze())
                pixel_values = FeatureExtractionDataset._processor_pixel_values(result)
                if pixel_values.ndim == 4 and pixel_values.shape[0] == 1:
                    pixel_values = pixel_values[0]
                if pixel_values.ndim != 3:
                    raise ValueError(
                        "Axial preprocessors must return [C,H,W] or [1,C,H,W], "
                        f"got {tuple(pixel_values.shape)}"
                    )
                processed_slices.append(pixel_values)
            return torch.stack(processed_slices, dim=1)

        result = self.preprocessor(tensor.unsqueeze(0))
        pixel_values = FeatureExtractionDataset._processor_pixel_values(result)
        if pixel_values.ndim == 5 and pixel_values.shape[0] == 1:
            pixel_values = pixel_values[0]
        elif pixel_values.ndim == 4:
            pass
        else:
            raise ValueError(
                "Volume preprocessors must return [C,D,H,W] or [1,C,D,H,W], "
                f"got {tuple(pixel_values.shape)}"
            )
        return pixel_values

    @staticmethod
    def _processor_pixel_values(result):
        if isinstance(result, dict) or hasattr(result, "__getitem__"):
            try:
                result = result["pixel_values"]
            except (KeyError, TypeError, IndexError):
                pass
        if not isinstance(result, torch.Tensor):
            result = torch.as_tensor(result)
        return result.to(dtype=torch.float32)

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
        crop_size_mm: Tuple[float, float, float] = (140.0, 140.0, 160.0),

        # Final model input
        # [Z,Y,X]
        target_size: Tuple[int, int, int] = (32, 128, 128),

        # Optional model voxel spacing in [X,Y,Z] millimeters.
        target_spacing_xyz: Optional[Tuple[float, float, float]] = None,

        # Padding outside the image
        pad_value: Optional[float] = None,
        modality: str = "ct",
        preprocessor=None,
        preprocessor_strategy: str = "volume",

        # Segmentation labels
        organ_label: int = 1,
        lesion_label: int = 2,

        # Minimum kidney component size in voxels
        min_organ_voxels: int = 1000,

        return_metadata: bool = True,

        **kwargs,
    ):
        super().__init__()

        modality = str(modality).lower()
        if modality not in {"ct", "mri"}:
            raise ValueError("modality must be 'ct' or 'mri'")
        if preprocessor_strategy not in {"axial", "volume"}:
            raise ValueError("preprocessor_strategy must be 'axial' or 'volume'")

        kwargs.pop("window_level", None)
        kwargs.pop("window_width", None)
        self.base_dataset = GenericRadiologyDataset(
            file_obj=file_obj,
            orientation=orientation,
            **kwargs,
        )
        self.orientation=orientation

        self.crop_size_mm = tuple(float(x) for x in crop_size_mm)

        self.target_size = tuple(int(x) for x in target_size)
        if len(self.crop_size_mm) != 3 or any(x <= 0 for x in self.crop_size_mm):
            raise ValueError("crop_size_mm must contain three positive values")
        if len(self.target_size) != 3 or any(x <= 0 for x in self.target_size):
            raise ValueError("target_size must contain three positive values")
        self.target_spacing_xyz = (
            None if target_spacing_xyz is None
            else tuple(float(x) for x in target_spacing_xyz)
        )
        if self.target_spacing_xyz is not None and (
            len(self.target_spacing_xyz) != 3 or any(x <= 0 for x in self.target_spacing_xyz)
        ):
            raise ValueError("target_spacing_xyz must contain three positive values")

        self.modality = modality
        self.pad_value = float((-1000.0 if modality == "ct" else 0.0) if pad_value is None else pad_value)
        self.preprocessor = preprocessor
        self.preprocessor_strategy = preprocessor_strategy

        self.organ_label = int(organ_label)

        self.lesion_label = int(lesion_label)

        self.min_organ_voxels = int(min_organ_voxels)

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
            (seg == self.organ_label) | (seg == self.lesion_label)
        )

        if not np.any(kidney_mask):
            return []

        # 26-connectivity is appropriate for anatomical structures.
        structure = np.ones((3, 3, 3), dtype=np.uint8,)

        labeled, n_components = label(kidney_mask, structure=structure)

        kidneys = []

        lesion_mask = (seg == self.lesion_label)

        for component_id in range(1, n_components + 1):

            component_mask = (labeled == component_id)

            volume = int(component_mask.sum())

            # Ignore tiny disconnected segmentation fragments.
            if volume < self.min_organ_voxels:
                continue

            coords = np.argwhere(component_mask)

            # coords are [z, y, x]
            centroid_zyx = coords.mean(axis=0)

            z, y, x = np.round(centroid_zyx).astype(int)

            lesion_volume = int(np.count_nonzero(lesion_mask & component_mask))

            kidneys.append(
                {
                    "component": component_id,
                    "center_xyz": (int(x), int(y), int(z)),
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
        kidneys.sort(key=lambda k: k["centroid_x"])

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
            kidneys = sorted(kidneys, key=lambda k: k["volume"], reverse=True)[:2]

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
        image: np.ndarray,
        center_xyz,
        spacing_xyz,
    ):
        """
        Crop a fixed physical region around one kidney.

        image:
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

        nx = int(np.round(size_x_mm / sx))

        ny = int(np.round(size_y_mm / sy))

        nz = int(np.round(size_z_mm / sz))

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
        px1 = max(0,x1 - image.shape[2])

        py0 = max(0, -y0)
        py1 = max(0, y1 - image.shape[1])

        pz0 = max(0, -z0)
        pz1 = max(0, z1 - image.shape[0])


        # Clamp to image.
        cx0 = max(0, x0)
        cx1 = min(image.shape[2], x1)


        cy0 = max(0, y0)
        cy1 = min(image.shape[1], y1)

        cz0 = max(0, z0)
        cz1 = min(image.shape[0], z1)

        crop = image[
            cz0:cz1,
            cy0:cy1,
            cx0:cx1,
        ]

        crop = np.pad(crop, ((pz0, pz1),
                             (py0, py1),
                             (px0, px1),),
                      mode="constant", constant_values=self.pad_value)

        return crop

    # ==============================================================
    # Resampling
    # ==============================================================

    def _resample_array(
        self,
        volume: np.ndarray,
        spacing_xyz=None,
    ):
        """
        Resample to configured spacing and size, returning output spacing [X,Y,Z].
        """

        target_d, target_h, target_w = self.target_size

        image = sitk.GetImageFromArray(volume.astype(np.float32))
        if spacing_xyz is not None:
            image.SetSpacing(tuple(spacing_xyz))

        original_size = image.GetSize()
        original_spacing = image.GetSpacing()

        if self.target_spacing_xyz is None:
            target_size_xyz = (target_w, target_h, target_d)
            target_spacing = tuple(
                original_spacing[axis] * original_size[axis] / target_size_xyz[axis]
                for axis in range(3)
            )
        else:
            target_spacing = self.target_spacing_xyz
            target_size_xyz = tuple(
                max(1, int(round((size - 1) * spacing / target_spacing[axis])) + 1)
                for axis, (size, spacing) in enumerate(zip(original_size, original_spacing))
            )

        resampler = sitk.ResampleImageFilter()

        resampler.SetSize(target_size_xyz)
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(self.pad_value)

        output = resampler.Execute(image)
        array = sitk.GetArrayFromImage(output).astype(np.float32)

        if self.target_spacing_xyz is not None:
            target_shape = (target_d, target_h, target_w)
            slices, padding = [], []
            for size, desired in zip(array.shape, target_shape):
                if size > desired:
                    start = (size - desired) // 2
                    slices.append(slice(start, start + desired))
                    padding.append((0, 0))
                else:
                    slices.append(slice(None))
                    total = desired - size
                    before = total // 2
                    padding.append((before, total - before))
            array = array[tuple(slices)]
            if any(before or after for before, after in padding):
                array = np.pad(array, padding, mode="constant", constant_values=self.pad_value)

        return array, tuple(output.GetSpacing())

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
        image,
        center_xyz,
        spacing_xyz,
        base_metadata,
        side,
    ):
        crop = self._crop_physical(
            image=image,
            center_xyz=center_xyz,
            spacing_xyz=spacing_xyz,
        )

        crop_preprocessor = getattr(self.preprocessor, "preprocess_kidney_crop", None)
        if crop_preprocessor is not None:
            result = crop_preprocessor(
                crop, spacing_xyz, base_metadata, self.modality, side
            )
            metadata_updates = {}
            if isinstance(result, dict):
                tensor = result["image"]
                metadata_updates = result.get("metadata", {})
            else:
                tensor = result
            tensor = torch.as_tensor(tensor, dtype=torch.float32)
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 4 or tensor.shape[0] != 1:
                raise ValueError(
                    "Kidney preprocessors must return [D,H,W] or [1,D,H,W], "
                    f"got {tuple(tensor.shape)}"
                )
            return tensor, {
                "preprocessor": type(self.preprocessor).__name__,
                "input_shape_zyx": tuple(crop.shape),
                "output_shape": tuple(tensor.shape),
                **metadata_updates,
            }

        processed, output_spacing = self._resample_array(crop, spacing_xyz)
        if self.preprocessor is None:
            if self.modality == "ct":
                processed = np.clip((processed + 1000.0) / 2000.0, 0.0, 1.0)
                intensity_transform = "clip_scale_-1000_1000_to_0_1"
                intensity_scale = "0_1"
            else:
                padding_mask = processed == self.pad_value
                valid = processed[~padding_mask]
                mean = float(valid.mean()) if valid.size else 0.0
                std = float(valid.std()) if valid.size else 1.0
                processed = (processed - mean) / max(std, 1e-8)
                processed[padding_mask] = 0.0
                intensity_transform = "per_volume_zscore"
                intensity_scale = "zscore"
        else:
            intensity_transform = "delegated_to_model_preprocessor"
            intensity_scale = "model_specific"

        tensor = self._to_tensor(processed)
        input_shape = tuple(processed.shape)
        if self.preprocessor is not None:
            tensor = FeatureExtractionDataset._apply_model_preprocessor(self, tensor)
        return tensor, {
            "preprocessor": type(self.preprocessor).__name__ if self.preprocessor else "modality_default",
            "input_shape_zyx": input_shape,
            "output_shape": tuple(tensor.shape),
            "spacing_xyz": output_spacing,
            "intensity_transform": intensity_transform,
            "intensity_scale": intensity_scale,
            "modality": self.modality,
            "intensity_min": float(tensor.min()),
            "intensity_max": float(tensor.max()),
            "intensity_mean": float(tensor.mean()),
            "intensity_std": float(tensor.std(unbiased=False)),
        }

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

        image = sample["image"][0]
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
                    "intensity_transform": "none",
                    "intensity_scale": "source_scale",
                    "pad_value": self.pad_value,
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

            image_width = image.shape[2]
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
            left_crop, left_preprocessing = self._process_kidney(
                image=image,
                center_xyz=left_kidney[
                    "center_xyz"
                ],
                spacing_xyz=spacing,
                base_metadata=base_metadata,
                side="left",
            )

        if right_kidney is not None:
            right_crop, right_preprocessing = self._process_kidney(
                image=image,
                center_xyz=right_kidney[
                    "center_xyz"
                ],
                spacing_xyz=spacing,
                base_metadata=base_metadata,
                side="right",
            )

        # ----------------------------------------------------------
        # Metadata
        # ----------------------------------------------------------

        metadata = {
            **base_metadata,
            "segmentation_available": True,
            "n_kidneys_detected": len(kidneys),
            "modality": self.modality,
            "crop_size_mm": self.crop_size_mm,
            "target_size_zyx": self.target_size,
            "target_spacing_xyz": self.target_spacing_xyz,
            #"left_preprocessing": left_preprocessing if left_kidney is not None else None,
            #"right_preprocessing": right_preprocessing if right_kidney is not None else None,
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
