import os
import sys
import importlib.util
import torch
from tqdm import tqdm
import pandas as pd
import SimpleITK as sitk
import numpy as np
from pathlib import Path
import cv2

from typing import List, Any, Optional, Union, Tuple

from transformers import AutoModel, AutoImageProcessor
from transformers.modeling_outputs import BaseModelOutputWithPooling

from torch.utils.data import Dataset, DataLoader

from utils import save_h5, match_segmentations
from dataset.generic import GenericTCIADataset
from dataset.tcia_features import KidneyFeatureDataset, kidney_collate_fn

"""
Need to make sure imports do not conflict with other modules.
This is useful for integrating other foundation models that may have their own dependencies and modules.
Is this really necessary?
_CT_DIR = Path(__file__).resolve().parent


def _load_local_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, _CT_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load local module '{filename}'")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_radiology_utils = _load_local_module("ct_radiology_utils", "utils.py")
_radiology_dataset = _load_local_module("ct_radiology_dataset", "dataset.py")

save_h5 = _radiology_utils.save_h5
GenericTCIADataset = _radiology_dataset.GenericTCIADataset
SegmentedTCIADataset = _radiology_dataset.SegmentedTCIADataset
"""

#from memory_profiler import profile

VALID_ORGANS = ['kidney']

class CTProcessor:
    def __init__(
        self,
        save_dir: str,
        worklist: str,
        fms: List[str],
        dataset_config: dict,
        ct_source: Optional[str],
        max_workers: Optional[int] = None,
        ct_extension: str = ".mha",
        nested_search: bool = True,
        use_segmentations: bool = False,
        segmentation_dir: Optional[str] = None,
        metadata_file: Optional[str] = "metadata.csv",
        renalclip_repo_path: Optional[str] = None,
        num_workers: int = 0,
        save_thumbnails: bool = True,
        organ: Optional[str] = None
    ) -> None:
        """
        Initialize a CT feature extraction pipeline from a worklist of CT files.

        Available methods:
            - `run_axial_feature_extraction_default`: extracts slice-level axial features
            - `run_axial_feature_extraction_kidney`: extracts kidney-specific axial features per side
            - `run_3d_feature_extraction_default`: extracts 3D volume features
            - `run_all`: runs feature extraction across the configured foundation models

        Parameters:
            save_dir (str):
                Root directory where extracted feature files and optional thumbnails are saved.
            worklist (str):
                Path to .csv file with paths to CT files -> generate with >python create_worklist.py /path/to/directory/
    
            
            fms (List):
                List of foundation model names to be used
                Currently support for: "curia", "tap-ct", "merlin"

            ct_source (str, optional):
                This is a path to the directory that holds the CT files, with the structure: 
                    ct_source/<patient_id>/<series_id>/<bar_code>.extension
                Add functionality to run create_worklist directly on this if worklist has not been generated yet?
            
            max_workers (int, optional):
                Maximum number of workers for data loading. If None, the default behavior will be used.
                Defaults to None.

            nested_search (bool):
                Search nested folders when looking for image files
            
            use_segmentations (bool):
                Whether to use segmentations for the CT volumes. If True, it will look for segmentation files 
                in the specified segmentation_dir and match them to the CT volumes based on the metadata file. 
                Defaults to False.

            dataset_config (dict, optional):
                Dictionary passed to the dataset constructor for loading and preprocessing CT volumes.
                Supported keys commonly include:
                - 'orientation' (str): image orientation, defaults to "LPS"
                - 'spatial_size' (int): target spatial size, defaults to 512
                - 'dtype' (type): output array dtype, defaults to np.float32
                - 'rel_pid_idx' (int): patient ID index relative to the file path root
                If None or omitted, dataset defaults are used.
            renalclip_repo_path (Optional[str], optional):
                Path to the RenalCLIP repository when using the "renalclip" foundation model. This is
                required for that model to import its local modules correctly.
            batch_size (int, optional):
                Number of CT samples to load per DataLoader batch. Defaults to 1.
            num_workers (int, optional):
                Number of worker processes used by the DataLoader for loading CT data. Defaults to 0.
            save_thumbnails (bool, optional):
                Whether to save center-slice thumbnail PNGs for processed CT volumes. Defaults to True.
            organ (Optional[str], optional):
                Target organ mode. Currently supports "kidney" for kidney-specific dataset handling.
                Unsupported values fall back to the default generic behavior.

        Returns:
            None: This method initializes the class instance and prepares the CT dataset and dataloader.

        Example
        -------
        Initialize the `CTProcessor` for a directory of CTs:

        >>> processor = CTProcessor(
        ...     save_dir="/data/pathology/projects/filip/TCGA-data/features/radiology/",
        ...     worklist="/data/pathology/projects/filip/TCGA-data/radiology_features/csv_files/ct_worklist.csv",
        ...     fms=["curia"],
        ...     dataset_config={'orientation': 'RAS', 'spatial_size': 256}
        ... )
        >>> print(f"CTProcessor initialized for {len(processor)} CT volumes.")


        Raises:
            AssertionError: If `ext` is not a list or if any extension does not start with a period.
        """
        self.worklist = pd.read_csv(worklist, header=None, skiprows=1)
       
        self.ct_source = ct_source
        self.ext = ct_extension
        self.save_dir = save_dir
        self.max_workers = max_workers
        #self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=0)
        self.fms = fms
        self.renalclip_repo_path = renalclip_repo_path
        self.save_thumbnails = save_thumbnails

        self.organ = organ
        if organ:
            if organ not in VALID_ORGANS:
                print(f"Warning: {organ} is not supported - using default behaviour")
                self.organ = None

        # Validate extension
        assert self.ext.startswith('.'), f'Invalid extension: {self.ext} (must start with a period)'

        if use_segmentations:
            metadata = pd.read_csv(os.path.join(ct_source, metadata_file))

            if not Path(segmentation_dir).is_absolute():
                segmentation_dir = os.path.join(ct_source, segmentation_dir)

            self.file_obj = match_segmentations(
                self.worklist[0].tolist(), metadata, segmentation_dir
            )
        else:
            self.file_obj=self.worklist

        if self.organ == 'kidney':
            self.dataset = KidneyFeatureDataset(
                file_obj=self.file_obj,
                **dataset_config
            )

            self.dataloader = DataLoader(self.dataset, batch_size=dataset_config['batch_size'], shuffle=False, num_workers=num_workers, collate_fn=kidney_collate_fn)
        else:
            self.dataset = GenericTCIADataset(
                file_obj=self.file_obj, 
                **dataset_config
            )

            self.dataloader = DataLoader(self.dataset, batch_size=dataset_config['batch_size'], shuffle=False, num_workers=num_workers)

    def __len__(self):
        return len(self.dataset)

    def capture_slice(self, ct_img, img_id):
        ct_img = ct_img.squeeze()
        middle = ct_img[int(ct_img.shape[0]/2), :, :].unsqueeze(-1).detach().cpu().numpy()
        middle = middle*255
        middle = middle.astype(int)
        save_path = f"{self.save_dir}/thumbnails/"
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        cv2.imwrite(os.path.join(save_path, f"{img_id}.png"), middle)
        

    def run_axial_feature_extraction_default(self,
                                     model,
                                     fmname,
                                     processor,
                                     device: torch.device,
                                     saveas: str = "h5"
                                     ):
        """
        Extract features from axial 2D slices of CT volumes, assuming that the input volume has the shape [B, S, H, W], 
        where B is the batch size, S is the number of axial slices, and H and W are the height and width of each slice. 
        The extracted features are saved to disk in either HDF5 or PyTorch format.
        """
        
        with torch.no_grad():
            loop = tqdm(self.dataloader)
            feature_dir = os.path.join(self.save_dir, f"axial_features_{fmname}")
            os.makedirs(feature_dir, exist_ok=True)
            broken_volumes = []

            for batch in loop:
                ct_images = batch["ct"] 

                for i, volume in enumerate(ct_images):
                    featureBlock = []
                    pid = batch['pid'][i]
                    sid = batch['sid'][i]
                    barcode = batch['barcode'][i]
                    if "metadata" in batch:
                        metadata = batch['metadata'][i]
                    if volume.ndim < 4 or any(s == 0 for s in volume.shape[-3:]):
                        loop.set_postfix_str(f'Warning: Volume has shape {volume.shape} for {pid} series {sid}. Skipping...')
                        broken_volumes.append((pid, sid[0]))
                        continue
                    # volume has the shape [B, D, H, W]
                    slices = volume.shape[1]

                    save_path = os.path.join(feature_dir, pid, f"{barcode}.{saveas}")
                    if os.path.exists(save_path):
                        loop.set_postfix_str(f'Features already extracted for {pid} series {sid}: {barcode}. Skipping...')
                        continue
                    for s in range(slices):
                        ct_slice = volume[:, s, :, :].to(device)
                        model_input = processor(ct_slice.squeeze()).to(device)
                        features = model(**model_input) 
                        featureBlock.append(features['pooler_output'])

                    if self.save_thumbnails:
                        self.capture_slice(volume, f"{pid}_{barcode}")
                    
                    # Concatenate features
                    featureBlock = torch.cat(featureBlock)

                    # Save the features to disk 
                    if saveas == 'h5':
                        save_h5(save_path=save_path,
                                assets = {
                                    'features' : featureBlock.detach().cpu().numpy(),
                                    'metadata': metadata
                                },
                                attributes = {
                                    'features': {'pid': pid, 'series': sid, 'savetodir': feature_dir, 'encoder': fmname},
                                },
                                mode='w')
                        loop.set_postfix_str(f'Features saved in h5 for {pid} series {sid}: {barcode}.')
                    elif saveas == 'pt':
                        torch.save(featureBlock, save_path)
                        loop.set_postfix_str(f'Features saved in pt for {pid} series {sid}: {barcode}.')
                    else:
                        raise ValueError(f'Invalid save_features_as: {saveas}. Only "h5" and "pt" are supported.')

            if len(broken_volumes) > 0:
                print(f"Warning: {len(broken_volumes)} volumes were skipped due to invalid shapes. See below for details:")
                for pid, sid in broken_volumes:
                    print(f" - Patient ID: {pid}, Series ID: {sid}")

    def run_3d_feature_extraction_default(
            self,
            model,
            fmname,
            processor,
            device: torch.device,
            saveas: str = "h5"
    ):
        """
        Extract features from 3D CT volumes. The input volume is expected to have the shape [B, D, H, W], 
        where B is the batch size, D is the depth (number of slices), and H and W are the height and width of each slice. 
        The extracted features are saved to disk in either HDF5 or PyTorch format.
        """

        with torch.no_grad():
            loop = tqdm(self.dataloader)
            feature_dir = os.path.join(self.save_dir, f"3d_features_{fmname}")
            os.makedirs(feature_dir, exist_ok=True)
            broken_volumes = []

            for batch in loop:
                ct_images = batch["ct"]

                for i, volume in enumerate(ct_images):
                    featureBlock = []
                    pid = batch['pid'][i]
                    sid = batch['sid'][i]
                    barcode = batch['barcode'][i]
                    if "metadata" in batch:
                        metadata = batch['metadata'][i]
                    if volume.ndim < 4 or any(s == 0 for s in volume.shape[-3:]):
                        loop.set_postfix_str(f'Warning: Volume has shape {volume.shape} for {pid} series {sid}. Skipping...')
                        broken_volumes.append((pid, sid[0]))
                        continue 

                    save_path = os.path.join(feature_dir, pid, f"{barcode}.{saveas}")
                    if os.path.exists(save_path):
                        loop.set_postfix_str(f'Features already extracted for {pid} series {sid}: {barcode}. Skipping...')
                        continue
                
                    if processor is not None:
                        volume = processor(volume)['pixel_values']

                    volume = volume.to(device)
                    output = model(volume)
                    if isinstance(output, BaseModelOutputWithPooling):
                        output = output.pooler_output

                    if self.save_thumbnails:
                        self.capture_slice(volume, f"{pid}_{barcode}")
                
                    # Save the features to disk 
                    if saveas == 'h5':
                        save_h5(save_path=save_path,
                                assets = {
                                    'features' : output.detach().cpu().numpy(),
                                    'metadata': metadata
                                },
                                attributes = {
                                    'features': {'pid': pid, 'series': sid, 'savetodir': feature_dir, 'encoder': fmname},
                                },
                                mode='w')
                        loop.set_postfix_str(f'Features saved in h5 for {pid} series {sid}: {barcode}.')
                    elif saveas == 'pt':
                        torch.save(featureBlock, save_path)
                        loop.set_postfix_str(f'Features saved in pt for {pid} series {sid}.')
                    else:
                        raise ValueError(f'Invalid save_features_as: {saveas}. Only "h5" and "pt" are supported.')

            if len(broken_volumes) > 0:
                print(f"Warning: {len(broken_volumes)} volumes were skipped due to invalid shapes. See below for details:")
                for pid, studyid in broken_volumes:
                    print(f" - Patient ID: {pid}, Series ID: {studyid}")

    def run_axial_feature_extraction_kidney(self,
                                         model,
                                         fmname,
                                         processor,
                                         device: torch.device,
                                         saveas: str = "h5"
                                         ):
        """
        Extract features from axial 2D slices of CT kidney volumes, assuming that the input volume has the shape [B, S, H, W], 
        where B is the batch size, S is the number of axial slices, and H and W are the height and width of each slice. 
        The extracted features are saved to disk in either HDF5 or PyTorch format.
        """
        
        with torch.no_grad():
            loop = tqdm(self.dataloader)
            feature_dir = os.path.join(self.save_dir, f"axial_features_{fmname}")
            os.makedirs(feature_dir, exist_ok=True)
            broken_volumes = []

            for batch in loop:

                for i, pid in enumerate(batch['pid']):
                    features = {}
                    feature_attrs = {}
                    sid = batch['sid'][i]
                    barcode = batch['barcode'][i]

                    if "metadata" in batch:
                        metadata = batch['metadata'][i]

                    save_path = os.path.join(feature_dir, pid, f"{barcode}.{saveas}")
                    if os.path.exists(save_path):
                        loop.set_postfix_str(f'Features already extracted for {pid} series {sid}: {barcode}. Skipping...')
                        continue

                    
                    for side in ['left', 'right']:
                        featureBlock = []
                        if batch[f"{side}_present"][i]:
                            volume = batch[side][i]
                            if volume.ndim < 4 or any(s == 0 for s in volume.shape[-3:]):
                                loop.set_postfix_str(f'Warning: Volume has shape {volume.shape} for {pid} series {sid}. Skipping...')
                                broken_volumes.append((pid, sid[0]))
                                continue
                            # volume has the shape [B, D, H, W]
                            slices = volume.shape[1]

                            for s in range(slices):
                                ct_slice = volume[:, s, :, :].to(device)
                                model_input = processor(ct_slice.squeeze()).to(device)
                                features = model(**model_input) 
                                featureBlock.append(features['pooler_output'])

                            if self.save_thumbnails:
                                self.capture_slice(volume, f"{pid}_{barcode}_{side}")
                        
                            # Concatenate features
                            featureBlock = torch.cat(featureBlock)
                            features[side] = featureBlock.detach().cpu().numpy()

                            feature_attrs[side] = {
                                'kidney_side': side,
                                'center_xyz': metadata.get(f"{side}_center_xyz"),
                                'volume_voxels': metadata.get(f"{side}_volume_voxels"),
                                'lesion_voxels': metadata.get(f"{side}_lesion_voxels")
                            }

                    if features:
                        
                        # Save the features to disk 
                        if saveas == 'h5':
                            save_h5(save_path=save_path,
                                    assets = {
                                        'features' : features
                                    },
                                    attributes = {'features': feature_attrs},
                                    root_attributes={
                                        "pid": pid,
                                        "sid": sid,
                                        "barcode": barcode,
                                        "savetodir": feature_dir,
                                        "encoder": fmname,
                                        "mode": "axial"
                                    },
                                    mode='w')
                            loop.set_postfix_str(f'Features saved in h5 for {pid} series {sid}: {barcode}.')
                        elif saveas == 'pt':
                            torch.save(features, save_path)
                            loop.set_postfix_str(f'Features saved in pt for {pid} series {sid}: {barcode}.')
                        else:
                            raise ValueError(f'Invalid save_features_as: {saveas}. Only "h5" and "pt" are supported.')

            if len(broken_volumes) > 0:
                print(f"Warning: {len(broken_volumes)} volumes were skipped due to invalid shapes. See below for details:")
                for pid, sid in broken_volumes:
                    print(f" - Patient ID: {pid}, Series ID: {sid}")
    
    def run_3d_feature_extraction_kidney(
            self,
            model,
            fmname,
            processor,
            device: torch.device,
            saveas: str = "h5"
    ):
        """
        Extract features from 3D CT volumes. The input volume is expected to have the shape [B, D, H, W], 
        where B is the batch size, D is the depth (number of slices), and H and W are the height and width of each slice. 
        The extracted features are saved to disk in either HDF5 or PyTorch format.
        """

        with torch.no_grad():
            loop = tqdm(self.dataloader)
            feature_dir = os.path.join(self.save_dir, f"3d_features_{fmname}")
            os.makedirs(feature_dir, exist_ok=True)
            broken_volumes = []

            for batch in loop:
            
                for i, pid in enumerate(batch['pid']):
                    features = {}
                    feature_attrs = {}
                    sid = batch['sid'][i]
                    barcode = batch['barcode'][i]

                    if "metadata" in batch:
                        metadata = batch['metadata'][i]

                    save_path = os.path.join(feature_dir, pid, f"{barcode}.{saveas}")
                    if os.path.exists(save_path):
                        loop.set_postfix_str(f'Features already extracted for {pid} series {sid}: {barcode}. Skipping...')
                        continue

                    for side in ['left', 'right']:

                        if batch[f"{side}_present"][i]:
                            volume = batch[side][i]
                            if volume.ndim < 4 or any(s == 0 for s in volume.shape[-3:]):
                                loop.set_postfix_str(f'Warning: Volume has shape {volume.shape} for {pid} series {sid}. Skipping...')
                                broken_volumes.append((pid, sid[0]))
                                continue
                            # Add channel dim if needed
                            if volume.ndim == 4:
                                volume = volume.unsqueeze(1)

                            if processor is not None:
                                volume = processor(volume)['pixel_values']

                            volume = volume.to(device)
                            
                            output = model(volume)
                            if isinstance(output, BaseModelOutputWithPooling):
                                output = output.pooler_output

                            if self.save_thumbnails:
                                self.capture_slice(volume, f"{pid}_{barcode}_{side}")
                        
                            
                            features[side] = output.detach().cpu().numpy()

                            feature_attrs[side] = {
                                'kidney_side': side,
                                'center_xyz': metadata.get(f"{side}_center_xyz"),
                                'volume_voxels': metadata.get(f"{side}_volume_voxels"),
                                'lesion_voxels': metadata.get(f"{side}_lesion_voxels")
                            }

                    if features:

                        # Save the features to disk 
                        if saveas == 'h5':
                            save_h5(save_path=save_path,
                                    assets = {
                                        'features' : features
                                    },
                                    attributes = {'features': feature_attrs},
                                    root_attributes={
                                        "pid": pid,
                                        "sid": sid,
                                        "barcode": barcode,
                                        "savetodir": feature_dir,
                                        "encoder": fmname,
                                        "mode": "axial"
                                    },
                                    mode='w')
                            loop.set_postfix_str(f'Features saved in h5 for {pid} series {sid}: {barcode}.')
                        elif saveas == 'pt':
                            torch.save(features, save_path)
                            loop.set_postfix_str(f'Features saved in pt for {pid} series {sid}: {barcode}.')
                        else:
                            raise ValueError(f'Invalid save_features_as: {saveas}. Only "h5" and "pt" are supported.')

            

            if len(broken_volumes) > 0:
                print(f"Warning: {len(broken_volumes)} volumes were skipped due to invalid shapes. See below for details:")
                for pid, studyid in broken_volumes:
                    print(f" - Patient ID: {pid}, Series ID: {studyid}")
            
    #@profile
    def run_all(self, device, saveas: str = "h5"):
        """
        Extract features using all specified foudantion models.
        Currently supports "curia", "tap-ct-2d", "tap-ct-3d" and "merlin".
        """
        for model_name in self.fms:
            if model_name == "curia":
                model = AutoModel.from_pretrained("raidium/curia")
                processor = AutoImageProcessor.from_pretrained("raidium/curia", trust_remote_code=True)
                mode = "axial"
                model.to(device)
            elif model_name == "curia-2":
                model = AutoModel.from_pretrained("raidium/curia-2")
                processor = AutoImageProcessor.from_pretrained("raidium/curia-2", trust_remote_code=True)
                mode = "axial"
                model.to(device)
            elif model_name == "tap-ct-2d":
                model = AutoModel.from_pretrained("fomofo/tap-ct-b-2d", trust_remote_code=True)
                processor = AutoImageProcessor.from_pretrained("fomofo/tap-ct-b-2d", trust_remote_code=True)
                mode = "axial"
                model.to(device)
            elif model_name == "tap-ct-3d":
                model = AutoModel.from_pretrained("fomofo/tap-ct-b-3d", trust_remote_code=True)
                processor = AutoImageProcessor.from_pretrained("fomofo/tap-ct-b-3d", trust_remote_code=True)
                mode = "3d"
                model.to(device)
            elif model_name in {"renalclip", "renal-clip"}:
                if self.renalclip_repo_path is None:
                    raise ValueError(
                        "renalclip_repo_path must be provided in CTProcessor when using the 'renalclip' foundation model."
                    )

                renalclip_repo_path = Path(self.renalclip_repo_path).expanduser().resolve()
                if not renalclip_repo_path.exists():
                    raise FileNotFoundError(
                        f"RenalCLIP repository path does not exist: {renalclip_repo_path}"
                    )

                repo_parent = renalclip_repo_path.parent
                for candidate in (str(repo_parent), str(renalclip_repo_path)):
                    if candidate not in sys.path:
                        sys.path.insert(0, candidate)

                for cache_key in (
                    "utils",
                    "utils.util",
                    "utils.logger",
                    "utils.data_util",
                    "models",
                    "models.RenalModel",
                    "RenalCLIP.models.RenalModel",
                ):
                    sys.modules.pop(cache_key, None)

                from RenalCLIP.models.RenalModel import RenalModel

                model = RenalModel(mode="finetune")
                model.eval()
                model.to(device)
                processor = None
                mode = "3d"
            elif model_name == "merlin":
                from merlin import Merlin
                model = Merlin(ImageEmbedding=True)
                model.eval()
                model.cuda()
                processor=None
                mode = "3d"
            else:
                raise NotImplementedError(f"{model_name} is not implemented")
            
            if mode == "axial":
                if self.organ == "kidney":
                    self.run_axial_feature_extraction_kidney(model=model, 
                                                             fmname=model_name, 
                                                             processor=processor, 
                                                             device=device, 
                                                             saveas=saveas)
                else:
                    self.run_axial_feature_extraction_default(model, model_name, processor, device, saveas=saveas)
            elif mode == "3d":
                if self.organ == "kidney":
                    self.run_3d_feature_extraction_kidney(model=model,
                                                          fmname=model_name,
                                                          processor=processor,
                                                          device=device,
                                                          saveas=saveas)
                else:
                    self.run_3d_feature_extraction_default(model, model_name, processor, device, saveas=saveas)

            print(f"Finished feature extraction for {model_name}. Features saved in {saveas} format in {self.save_dir}.")
            # After first run, toggle save_thumbnails off
            self.save_thumbnails = False


