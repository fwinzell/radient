# datasets/generic_tcia.py

import os
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import SimpleITK as sitk
from torch.utils.data import Dataset

import warnings

class GenericRadiologyDataset(Dataset):
    """
    Modality-neutral loader for 3D medical images with optional segmentations.
    Supports DICOM series directories and single-file formats (e.g., NIfTI, MHA, etc.).
    Check SimpleITK documentation for supported formats: https://simpleitk.readthedocs.io/en/master/link_ImageFileReader.html
    

    file_obj:
        DataFrame:
            column 0 -> image path or DICOM series directory
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
        patient_id_column: Optional[Union[str, int]] = None,
        study_id_column: Optional[Union[str, int]] = None,
        series_id_column: Optional[Union[str, int]] = None,
        dicom_series: bool = True,
        image_path_column: Optional[Union[str, int]] = None,
        segmentation_path_column: Optional[Union[str, int]] = None,
        modality_column: Optional[Union[str, int]] = None,
        sequence_column: Optional[Union[str, int]] = None,
        paired_image_path_column: Optional[Union[str, int]] = None,
        *args,
        **kwargs,
    ):
        self.orientation = orientation
        self.dtype = dtype
        self.rel_pid_idx = rel_pid_idx
        self.segmentation_column = segmentation_column
        self.allow_missing_segmentations = allow_missing_segmentations
        self.spatial_size = spatial_size
        self.dicom_series = dicom_series
        self.modalities = self._column_values(file_obj, modality_column)
        self.sequences = self._column_values(file_obj, sequence_column)
        self.paired_paths = self._column_values(file_obj, paired_image_path_column)
        self.patient_ids = self._column_values(file_obj, patient_id_column)
        self.study_ids = self._column_values(file_obj, study_id_column)
        self.series_ids = self._column_values(file_obj, series_id_column)

        if isinstance(file_obj, pd.DataFrame):
            if file_obj.shape[1] == 0:
                raise ValueError("file_obj has no columns")

            self.paths = self._get_column(file_obj, image_path_column, 0)

            if (
                segmentation_path_column is not None
            ):
                self.seg_paths = self._get_column(file_obj, segmentation_path_column, segmentation_column)
            elif segmentation_column is not None and file_obj.shape[1] > segmentation_column:
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
            raise ValueError("No image paths provided")

        if len(self.paths) != len(self.seg_paths):
            raise ValueError(
                "CT and segmentation path lists have different lengths"
            )

        self.root_dir = Path(
            os.path.commonpath([str(p) for p in self.paths])
        )

    @staticmethod
    def _get_column(frame, column, default_index):
        if column is None:
            return frame.iloc[:, default_index].tolist()
        try:
            return (frame[column] if isinstance(column, str) else frame.iloc[:, column]).tolist()
        except (KeyError, IndexError, TypeError):
            raise ValueError(f"Input column not found: {column}")

    @staticmethod
    def _column_values(file_obj, column):
        if column is None or not isinstance(file_obj, pd.DataFrame):
            return None
        try:
            values = file_obj[column] if isinstance(column, str) else file_obj.iloc[:, column]
        except (KeyError, IndexError, TypeError):
            raise ValueError(f"Identifier column not found: {column}")
        return values.tolist()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_sitk(self, path: Union[str, Path]) -> sitk.Image:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")

        if path.is_dir():
            if not self.dicom_series:
                raise ValueError(f"Directory input requires dicom_series=True: {path}")
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(path))
            if not series_ids:
                raise ValueError(f"No readable DICOM series found in: {path}")
            if len(series_ids) != 1:
                raise ValueError(
                    f"Found {len(series_ids)} DICOM series in {path}; pass a series directory "
                    "or a manifest path that identifies one series"
                )
            filenames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(path), series_ids[0])
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(filenames)
            return reader.Execute()
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
            if not rel.parts:
                return None
            return rel.parts[self.rel_pid_idx]
        except (ValueError, IndexError):
            return None

    def __len__(self):
        return len(self.paths)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def __getitem__(self, index):

        if not isinstance(index, (int, np.integer)):
            raise TypeError(f"Dataset index must be an integer, got {type(index).__name__}: {index!r}")
        if index < 0 or index >= len(self):
            raise IndexError(f"Dataset index {index} out of range for {len(self)} samples")

        vol_path = Path(self.paths[index])

        pid = self.patient_ids[index] if self.patient_ids is not None else self.get_patient_id(vol_path)
        sid = self.study_ids[index] if self.study_ids is not None else vol_path.parent.name
        barcode = self.series_ids[index] if self.series_ids is not None else (vol_path.stem if vol_path.is_file() else vol_path.name)

        # -------------------------
        # 3D Image volume
        # -------------------------
        img_sitk = self._load_and_orient(
            vol_path,
            is_segmentation=False,
        )

        img_np = self._to_numpy(
            img_sitk,
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
                    if seg.shape != img_np.shape:
                        print("Warning, shape mismatch detected. Attempting to adjust...")
                        seg = self._check_dimensions(seg)
                        img_np = self._check_dimensions(img_np)
                        if seg.shape != img_np.shape:
                            if not self.allow_missing_segmentations:
                                raise ValueError(
                                    f"{barcode}: CT and segmentation sizes differ: "
                                    f"{img_np.shape} vs {seg.shape}"
                                )
                            else:
                                warnings.warn(f"{barcode}: CT and segmentation sizes differ: {img_np.shape} vs {seg.shape} \n Setting seg to None")
                                seg = None


                elif not self.allow_missing_segmentations:
                    raise FileNotFoundError(
                        f"{barcode}: segmentation not found: {seg_path}"
                    )

        metadata = {
            "spacing": img_sitk.GetSpacing(),       # x,y,z in mm
            "origin": img_sitk.GetOrigin(),
            "direction": img_sitk.GetDirection(),
            "shape_zyx": img_np.shape,
            "has_segmentation": seg is not None,
            "modality": self.modalities[index] if self.modalities is not None else None,
            "sequence": self.sequences[index] if self.sequences is not None else None,
            "paired_image_path": self.paired_paths[index] if self.paired_paths is not None else None,
            "source_path": str(vol_path),
        }

        return {
            "pid": pid,
            "sid": sid,
            "barcode": barcode,
            "image": img_np,
            "ct": img_np,  # Backward compatibility for existing CT feature pipelines. Remove in future versions.
            "seg": seg,
            "metadata": metadata,
        }
