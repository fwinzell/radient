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
        if isinstance(value, dict):
            value = json.dumps(value)
        elif value is None:
            value = "None"
        elif isinstance(value, tuple):
            value = np.asarray(value)

        try:
            h5_object.attrs[key] = value
        except Exception as e:
            raise RuntimeError(
                f"Failed to save HDF5 attribute '{key}'={value!r}"
            ) from e


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

                    if isinstance(feature, torch.Tensor):
                        feature = feature.detach().cpu().numpy()
                    elif not isinstance(feature, np.ndarray):
                        feature = np.asarray(feature)

                    if fkey in group:
                        del group[fkey]

                    dset = group.create_dataset(fkey, data=feature)

                    # Explicitly save feature-specific attributes
                    feature_attributes = group_attributes.get(fkey, {})

                    _save_attributes(dset, feature_attributes)
                continue

            # --------------------------------------------------------
            # Normal, non-nested asset
            # --------------------------------------------------------
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            elif not isinstance(value, np.ndarray):
                value = np.asarray(value)

            if key in f:
                del f[key]

            dset = f.create_dataset(key, data=value)

            if attributes is not None:
                _save_attributes(dset, attributes.get(key))


def match_segmentations(worklist, metadata, seg_dir):
    #metadata = pd.read_csv(os.path.join(main_dir, "metadata.csv"))
    lookup = metadata.set_index("File Location")["Series UID"]

    root_dir = Path(os.path.commonpath(worklist))

    #rel_path = Path(worklist[0]).relative_to(root_dir).parent.parent
    #file_path = "./" + root_dir.name.replace("-mha", "") + "/" + str(rel_path)

    study_uids = [Path(p).parent.name for p in worklist]
    seg_files = [p for p in Path(seg_dir).iterdir() if p.suffix == ".nrrd"]
    #count = sum(1 for p in seg_files if p.name.startswith("KIRP_"))

    rows = []
    for j,full_path in enumerate(worklist):
        rel_path = Path(full_path).relative_to(root_dir).parent.parent
        file_path = "./" + root_dir.name.replace("-mha", "") + "/" + str(rel_path)
        uid = lookup[file_path]

        seg_id = uid.replace(".", "_").replace("-0", "")
        idx = [i for i, seg in enumerate(seg_files) if seg_id in seg.stem]

        if len(idx) == 0:
            print(f"Warning, no segmentation found for {uid}")
            #pairs[uid] = (worklist[j], None)
            continue
        elif len(idx) > 1:
            print(f"Warning, multiple segmentations found for {uid}: {idx}")
        #pairs[uid] = (worklist[j],seg_files[idx[0]])
        rows.append((worklist[j], seg_files[idx[0]]))
        

    return pd.DataFrame(rows)


    
def valid_segmentation(self, seg_path):
    seg = sitk.ReadImage(seg_path)
    seg = sitk.GetArrayFromImage(seg)
    is_non_zero = np.any(seg)
    return is_non_zero

if __name__ == "__main__":
    worklist = pd.read_csv("/Users/filipwinzell/Python/Radboud/tcga_kirp.csv", header=None, skiprows=1)[0].tolist()
    metadata = pd.read_csv("/Volumes/oncology/experiments/sarah_kidney_abnormality_segmentation/datasets/TCGA-KIRP/metadata.csv")   
    seg_dir = "/Volumes/oncology/experiments/sarah_kidney_abnormality_segmentation/datasets/TCGA-KIRC/onco_segmentations"
    file_obj =match_segmentations(worklist, metadata, seg_dir)
    print(file_obj.head())
