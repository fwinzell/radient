# datasets/generic_tcia.py

import os
from pathlib import Path
from typing import Iterable, Optional, Union, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

import warnings

class GenericTCIADataset(Dataset):
    """
    Generic loader for CT volumes with optional segmentations.

    file_obj:
        DataFrame:
            column 0 -> CT path
            column 1 -> segmentation path (optional)

    The generic dataset:
        - loads the volume
        - optionally loads segmentation
        - reorients both to the requested orientation
        - returns them in NumPy [D, H, W] convention
        - extracts metadata if available
    """

    def __init__(
        self,
        file_obj: Union[pd.DataFrame, Iterable[str]],
        orientation: str = "RAS",
        dtype: type = np.float32,
        rel_pid_idx: int = 0,
        segmentation_column: Optional[int] = 1,
        allow_missing_segmentations: bool = True,
        spatial_size: int = 512,
        *args,
        **kwargs,
    ):
        self.orientation = orientation
        self.dtype = dtype
        self.rel_pid_idx = rel_pid_idx
        self.segmentation_column = segmentation_column
        self.allow_missing_segmentations = allow_missing_segmentations
        self.spatial_size = spatial_size

        if isinstance(file_obj, pd.DataFrame):
            if file_obj.shape[1] == 0:
                raise ValueError("file_obj has no columns")

            self.paths = file_obj.iloc[:, 0].tolist()

            if (
                segmentation_column is not None
                and file_obj.shape[1] > segmentation_column
            ):
                self.seg_paths = file_obj.iloc[:, segmentation_column].tolist()
            else:
                self.seg_paths = [None] * len(self.paths)

        elif isinstance(file_obj, Iterable) and not isinstance(file_obj, (str, bytes)):
            self.paths = list(file_obj)
            self.seg_paths = [None] * len(self.paths)

        else:
            raise TypeError(
                "file_obj must be a pandas DataFrame or an iterable of paths"
            )

        if len(self.paths) == 0:
            raise ValueError("No CT paths provided")

        if len(self.paths) != len(self.seg_paths):
            raise ValueError(
                "CT and segmentation path lists have different lengths"
            )

        self.root_dir = Path(
            os.path.commonpath([str(p) for p in self.paths])
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_sitk(self, path: Union[str, Path]) -> sitk.Image:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")

        return sitk.ReadImage(str(path))

    def _load_and_orient(
        self,
        path: Union[str, Path],
        is_segmentation: bool = False,
    ) -> sitk.Image:
        """
        Reorient without interpolation.

        DICOMOrient only permutes/flips axes, so it is safe for both
        CT and segmentation labels.
        """
        img = self._load_sitk(path)

        if img.GetDimension() != 3:
            raise ValueError(
                f"Expected 3D volume, got {img.GetDimension()}D: {path}"
            )

        img = sitk.DICOMOrient(
            img,
            self.orientation,
        )

        return img

    def _check_dimensions(self, vol):
        """
        Ensures that the volume is returned as [slice_axis, spatial_axes]
        If a non_spatial dimension cannot be determined, it does nothing
        Useful for doing 2D ooperations on 3D volumes

        raises ValueError if a non-3D volume is provided
        """

        if vol.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {tuple(vol.shape)}")

        shape = list(vol.shape)

        # Find the axis that is NOT the spatial size
        non_spatial_axes = [i for i, d in enumerate(shape) if d != self.spatial_size]

        if len(non_spatial_axes) == 1:
            slice_axis = non_spatial_axes[0]
        elif len(non_spatial_axes) == 2:
            slice_axis = np.argmin(vol.shape)
        else:
            slice_axis = 0

        # Move slice axis to the front
        if slice_axis == 0:
            return vol
        elif slice_axis == 1:
            vol = np.transpose(vol, (1,0,2)) #vol.permute(1, 0, 2)
        else:  
            vol = np.transpose(vol, (2,0,1)) #vol = vol.permute(2, 0, 1)
        return vol

    @staticmethod
    def _to_numpy(
        img: sitk.Image,
        dtype: type = np.float32,
    ) -> np.ndarray:
        """
        SimpleITK -> NumPy gives [D, H, W] = [Z, Y, X].
        """
        arr = sitk.GetArrayFromImage(img)

        return arr.astype(dtype, copy=False)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_patient_id(
        self,
        path: Union[str, Path],
    ) -> Optional[str]:
        """
        Return patient id from the path
        """
        
        path = Path(path)

        try:
            rel = path.relative_to(self.root_dir)
            return rel.parts[self.rel_pid_idx]
        except ValueError:
            return None

    def __len__(self):
        return len(self.paths)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def __getitem__(self, index):

        ct_path = Path(self.paths[index])

        pid = self.get_patient_id(ct_path)
        sid = ct_path.parent.name
        barcode = ct_path.stem

        # -------------------------
        # CT
        # -------------------------
        ct_img = self._load_and_orient(
            ct_path,
            is_segmentation=False,
        )

        ct = self._to_numpy(
            ct_img,
            dtype=self.dtype,
        )

        # -------------------------
        # Optional segmentation
        # -------------------------
        seg = None
        seg_img = None
        seg_path = self.seg_paths[index]

        if seg_path is not None:

            # Handle NaN / empty path from pandas
            if isinstance(seg_path, float) and np.isnan(seg_path):
                seg_path = None

            if seg_path is not None and str(seg_path).strip() != "":
                seg_path = Path(seg_path)

                if seg_path.exists():
                    seg_img = self._load_and_orient(
                        seg_path,
                        is_segmentation=True,
                    )

                    seg = self._to_numpy(
                        seg_img,
                        dtype=np.int16,
                    )

                    # Make sure seg and ct have the same shape and are aligned
                    if seg.shape != ct.shape:
                        print("Warning, shape mismatch detected. Attempting to adjust...")
                        seg = self._check_dimensions(seg)
                        ct = self._check_dimensions(ct)
                        if seg.shape != ct.shape:
                            if not self.allow_missing_segmentations:
                                raise ValueError(
                                    f"{barcode}: CT and segmentation sizes differ: "
                                    f"{ct.shape} vs {seg.shape}"
                                )
                            else:
                                warnings.warn(f"{barcode}: CT and segmentation sizes differ: {ct.shape} vs {seg.shape} \n Setting seg to None")
                                seg = None


                elif not self.allow_missing_segmentations:
                    raise FileNotFoundError(
                        f"{barcode}: segmentation not found: {seg_path}"
                    )

        metadata = {
            "spacing": ct_img.GetSpacing(),       # x,y,z in mm
            "origin": ct_img.GetOrigin(),
            "direction": ct_img.GetDirection(),
            "shape_zyx": ct.shape,
            "has_segmentation": seg is not None,
        }

        return {
            "pid": pid,
            "sid": sid,
            "barcode": barcode,
            "ct": ct,
            "seg": seg,
            "metadata": metadata,
        }