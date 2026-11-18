from pathlib import Path

import h5py
import json
import os
import pandas as pd
import numpy as np
import re

SUPPORTED_EXTENSIONS = [".mha", ".nii", ".tif", ".tiff", ".dcm"]

import os
import json
import h5py
import numpy as np
import torch


def _save_attributes(h5_object, attributes):
    if attributes is None:
        return

    for key, value in attributes.items():
        if isinstance(value, np.str_):
            value = str(value)
        if isinstance(value, dict):
            value = json.dumps(value, default=_json_default)
        elif value is None:
            value = "None"
        elif isinstance(value, tuple):
            value = np.asarray(value)
        if isinstance(value, np.ndarray) and value.dtype.kind == "U":
            value = value.astype(h5py.string_dtype(encoding="utf-8"))

        try:
            h5_object.attrs[key] = value
        except Exception as e:
            raise RuntimeError(
                f"Failed to save HDF5 attribute '{key}'={value!r}"
            ) from e


def _json_default(value):
    """Convert NumPy scalar values to JSON-compatible Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _create_dataset(h5_group, key, value):
    """Create a dataset while converting NumPy Unicode strings for h5py."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    elif not isinstance(value, np.ndarray):
        value = np.asarray(value)

    if value.dtype.kind == "U":
        if value.ndim == 0:
            value = str(value.item())
            return h5_group.create_dataset(
                key, data=value, dtype=h5py.string_dtype(encoding="utf-8")
            )
        value = value.astype(object)
        return h5_group.create_dataset(
            key, data=value, dtype=h5py.string_dtype(encoding="utf-8")
        )
    if value.dtype.kind == "O":
        flat = value.reshape(-1)
        if all(isinstance(item, (str, bytes, np.str_, np.bytes_)) for item in flat):
            value = np.asarray(
                [item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item) for item in flat],
                dtype=object,
            ).reshape(value.shape)
            return h5_group.create_dataset(
                key, data=value, dtype=h5py.string_dtype(encoding="utf-8")
            )
    return h5_group.create_dataset(key, data=value)


def save_h5(
        save_path, 
        assets, 
        attributes = None,
        root_attributes = None, 
        mode = 'w'):
    """
    The `save_h5` function saves a dictionary of assets to an HDF5 file. This is commonly used to store 
    large datasets or hierarchical data structures in a compact and organized format.

    Nested dictionaries in `assets` are represented as HDF5 groups.

    Parameters:
    -----------
    save_path : str
        The path where the HDF5 file will be saved.
    assets : dict
        A dictionary containing the data to save. Keys represent dataset names, and values are NumPy arrays.
    attributes : dict, optional
        A dictionary mapping dataset names to additional dataset-specific metadata (attributes) to save alongside the data. Defaults to None.
    root_attributes : dict, optional
        Attributes of the .h5 file itself
        Example:
            {
                "pid": "TCGA-AA-0001"
            }
    mode : str, optional
        The file mode for opening the HDF5 file. Options include 'w' (write) and 'a' (append). Defaults to 'w'.

    Returns:
    --------
    None
        The function writes data and attributes to the specified HDF5 file.

    Example:
    --------
    >>> assets = {"features": features}

    >>> attributes = {
            "features": {
                side: {
                    "kidney_side": side,
                    "center_xyz": metadata.get(f"{side}_center_xyz"),
                    "kidney_volume_voxels": metadata.get(f"{side}_volume_voxels"),
                    "lesion_volume_voxels": metadata.get(f"{side}_lesion_voxels"),
                }
                for side in features
            }
        }

    >>> root_attributes = {
            "pid": pid,
            "series": sid,
            "barcode": barcode,
            "encoder": fmname}

    >>> save_h5(
            save_path=save_path,
            assets=assets,
            attributes=attributes,
            root_attributes=root_attributes,
            mode="w",
            )
    
    """

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with h5py.File(save_path, mode) as f:
        _save_attributes(f, root_attributes)

        for key, value in assets.items():
            if value is None:
                continue

            # --------------------------------------------------------
            # Nested feature dictionary
            # --------------------------------------------------------
            if isinstance(value, dict):

                group = f.require_group(key)

                group_attributes = (attributes.get(key, {}) if attributes is not None else {})

                for fkey, feature in value.items():

                    if feature is None:
                        continue

                    if fkey in group:
                        del group[fkey]

                    dset = _create_dataset(group, fkey, feature)

                    # Explicitly save feature-specific attributes
                    feature_attributes = group_attributes.get(fkey, {})

                    _save_attributes(dset, feature_attributes)
                continue

            # --------------------------------------------------------
            # Normal, non-nested asset
            # --------------------------------------------------------
            if key in f:
                del f[key]

            dset = _create_dataset(f, key, value)

            if attributes is not None:
                _save_attributes(dset, attributes.get(key))



