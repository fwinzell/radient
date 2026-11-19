"""Load Merlin with its documented CT volume preprocessing."""

import numpy as np
import SimpleITK as sitk
import torch


class MerlinPreprocessor:
    """Match Merlin's CT orientation, spacing, intensity and shape transforms."""

    def __init__(self, config=None):
        config = config or {}
        dataset_config = config.get("dataset", {})
        self.target_spacing_xyz = tuple(config.get("target_spacing_xyz", (1.5, 1.5, 3.0)))
        self.target_shape_zyx = tuple(config.get("target_shape_zyx", (160, 224, 224)))
        self.kidney_target_spacing_xyz = tuple(
            config.get(
                "kidney_target_spacing_xyz",
                dataset_config.get("target_spacing_xyz", self.target_spacing_xyz),
            )
        )
        self.kidney_target_shape_zyx = tuple(
            config.get(
                "kidney_target_shape_zyx",
                dataset_config.get("target_size", self.target_shape_zyx),
            )
        )

    def _resample_and_shape(self, image, target_spacing_xyz, target_shape_zyx):
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()
        output_size = tuple(
            max(1, int(round((size - 1) * spacing / target_spacing)) + 1)
            for size, spacing, target_spacing in zip(
                original_size, original_spacing, target_spacing_xyz
            )
        )
        resampled = sitk.Resample(
            image,
            output_size,
            sitk.Transform(),
            sitk.sitkLinear,
            image.GetOrigin(),
            target_spacing_xyz,
            image.GetDirection(),
            -1000.0,
            sitk.sitkFloat32,
        )
        array = sitk.GetArrayFromImage(resampled)
        array = np.clip((array + 1000.0) / 2000.0, 0.0, 1.0)

        slices, padding = [], []
        for size, desired in zip(array.shape, target_shape_zyx):
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
            array = np.pad(array, padding, mode="constant", constant_values=0.0)
        return array, resampled

    def preprocess_dataset(self, volume, metadata, modality):
        if modality != "ct":
            raise ValueError("Merlin preprocessing currently supports CT only")

        image = sitk.GetImageFromArray(np.asarray(volume, dtype=np.float32))
        image.SetSpacing(tuple(metadata["spacing"]))
        image.SetOrigin(tuple(metadata["origin"]))
        image.SetDirection(tuple(metadata["direction"]))
        image = sitk.DICOMOrient(image, "RAS")
        array, resampled = self._resample_and_shape(
            image, self.target_spacing_xyz, self.target_shape_zyx
        )
        return {
            "image": torch.from_numpy(np.ascontiguousarray(array[None])).float(),
            "metadata": {
                "spacing": self.target_spacing_xyz,
                "origin": resampled.GetOrigin(),
                "direction": resampled.GetDirection(),
                "shape_zyx": tuple(array.shape),
                "orientation": "RAS",
                "intensity_transform": "clip_scale_-1000_1000_to_0_1",
                "spatial_transform": "resample_1.5x1.5x3mm_center_crop_or_pad_224x224x160",
                "intensity_scale": "0_1",
                "pad_value": 0.0,
            },
        }

    def preprocess_kidney_crop(self, volume, spacing_xyz, metadata, modality, side):
        if modality != "ct":
            raise ValueError("Merlin preprocessing currently supports CT only")
        image = sitk.GetImageFromArray(np.asarray(volume, dtype=np.float32))
        image.SetSpacing(tuple(spacing_xyz))
        image.SetOrigin((0.0, 0.0, 0.0))
        image.SetDirection(tuple(metadata["direction"]))
        image = sitk.DICOMOrient(image, "RAS")
        array, resampled = self._resample_and_shape(
            image,
            self.kidney_target_spacing_xyz,
            self.kidney_target_shape_zyx,
        )
        return {
            "image": torch.from_numpy(np.ascontiguousarray(array[None])).float(),
            "metadata": {
                "spacing_xyz": self.kidney_target_spacing_xyz,
                "shape_zyx": tuple(array.shape),
                "orientation": "RAS",
                "intensity_transform": "clip_scale_-1000_1000_to_0_1",
                "intensity_scale": "0_1",
                "spatial_transform": "resample_then_center_crop_or_pad",
                "pad_value": 0.0,
                "kidney_side": side,
            },
        }


def load_merlin(config, device):
    from merlin import Merlin

    model = Merlin(**config.get("init_kwargs", {"ImageEmbedding": True}))
    model.to(device)
    model.eval()
    return model, MerlinPreprocessor(config)
