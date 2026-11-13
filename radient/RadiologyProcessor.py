import os
import torch
from tqdm import tqdm
import pandas as pd
import SimpleITK as sitk
import numpy as np
from pathlib import Path
import cv2

from typing import List, Any, Optional, Union, Callable, Type

from transformers.modeling_outputs import BaseModelOutputWithPooling

from torch.utils.data import DataLoader

from utils import save_h5
from radient.dataset.generic import GenericRadiologyDataset
from radient.dataset.features import FeatureExtractionDataset
from radient.model_zoo import load_foundation_model

class RadiologyProcessor:
    def __init__(
        self,
        save_dir: str,
        worklist: Union[str, os.PathLike, pd.DataFrame],
        fms: List[str],
        dataset_config: Optional[dict] = None,
        max_workers: Optional[int] = None,
        image_extension: str = ".mha",
        nested_search: bool = True,
        use_segmentations: bool = False,
        allow_missing_segmentations: bool = False,
        model_configs: Optional[dict] = None,
        num_workers: int = 0,
        save_thumbnails: bool = True,
        batch_size: int = 1,
        dataset_class: Optional[Type] = None,
        collate_fn: Optional[Callable] = None,
    ) -> None:
        """
        Initialize a radiology feature extraction pipeline from a worklist.

        Available methods:
            - `run_axial_feature_extraction_default`: extracts slice-level axial features
            - `run_3d_feature_extraction_default`: extracts 3D volume features
            - `run_all`: runs feature extraction across the configured foundation models

        Parameters:
            save_dir (str):
                Root directory where extracted feature files and optional thumbnails are saved.
            worklist (str):
                CSV path or DataFrame. With ``use_segmentations=True``, the first
                column must contain volume paths and the second the matching
                segmentation paths, with one pair per row.
    
            
            fms (List):
                List of foundation model names to be used
                Currently support for: "curia", "tap-ct", "merlin"

            max_workers (int, optional):
                Maximum number of workers for data loading. If None, the default behavior will be used.
                Defaults to None.

            nested_search (bool):
                Search nested folders when looking for image files
            
            use_segmentations (bool):
                Whether to use segmentation-aware preprocessing. If True, the
                worklist must have a second column with segmentation paths.
                Defaults to False.
            allow_missing_segmentations (bool):
                When using segmentations, require a non-empty, readable mask
                for every volume. If True, missing masks are allowed and the
                feature dataset applies its configured no-segmentation fallback.
                Defaults to False.

            dataset_config (dict, optional):
                Dictionary passed to the selected dataset constructor.
                Supported keys commonly include:
                - 'orientation' (str): image orientation, defaults to "LPS"
                - 'spatial_size' (int): target spatial size, defaults to 512
                - 'dtype' (type): output array dtype, defaults to np.float32
                - 'rel_pid_idx' (int): patient ID index relative to the file path root
                If None or omitted, dataset defaults are used.
            model_configs (dict, optional): Model-specific options, keyed by model
                name. For example: ``{"renalclip": {"repo_path": "/path/to/RenalCLIP"}}``.
            batch_size (int, optional):
                Number of image samples to load per DataLoader batch. Defaults to 1.
            num_workers (int, optional):
                Number of worker processes used by the DataLoader. Defaults to 0.
            save_thumbnails (bool, optional):
                Whether to save center-slice thumbnail PNGs for processed CT volumes. Defaults to True.
            dataset_class: Optional custom Dataset class. It receives `file_obj`
                and the entries in `dataset_config` as constructor arguments.
            collate_fn: Optional custom DataLoader collate function, for example
                a function paired with `dataset_class`.

        Returns:
            None: Initializes the dataset and dataloader.

        Example
        -------
        Initialize the `RadiologyProcessor`:

        >>> processor = RadiologyProcessor(
        ...     save_dir="/data/pathology/projects/filip/TCGA-data/features/radiology/",
        ...     worklist="/data/pathology/projects/filip/TCGA-data/radiology_features/csv_files/ct_worklist.csv",
        ...     fms=["curia"],
        ...     dataset_config={'orientation': 'RAS', 'spatial_size': 256}
        ... )
        >>> print(f"RadiologyProcessor initialized for {len(processor)} volumes.")


        Raises:
            AssertionError: If `ext` is not a list or if any extension does not start with a period.
        """
        if not save_dir:
            raise ValueError("save_dir is required")
        if not fms:
            raise ValueError("fms must contain at least one foundation model")

        if use_segmentations and not isinstance(worklist, pd.DataFrame):
            if Path(worklist).suffix.lower() != ".csv":
                raise ValueError(
                    "use_segmentations=True requires a .csv worklist with volume "
                    "paths in column 1 and matching segmentation paths in column 2."
                )

        if isinstance(worklist, pd.DataFrame):
            self.worklist = worklist.copy()
        else:
            self.worklist = pd.read_csv(worklist)
        if self.worklist.empty:
            raise ValueError("worklist contains no rows")
        dataset_config = dict(dataset_config or {})
       
        self.image_extension = image_extension
        self.save_dir = save_dir
        self.max_workers = max_workers
        #self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=0)
        self.fms = fms
        self.model_configs = model_configs or {}
        self.save_thumbnails = save_thumbnails

        # Validate extension
        assert self.image_extension.startswith('.'), f'Invalid extension: {self.image_extension} (must start with a period)'

        self.file_obj = self.worklist
        self.use_segmentations = bool(use_segmentations)
        self.allow_missing_segmentations = bool(allow_missing_segmentations)

        if self.use_segmentations:
            if self.worklist.shape[1] < 2:
                raise ValueError(
                    "use_segmentations=True requires a worklist CSV with at least "
                    "two columns: volume path first, matching segmentation path second."
                )
            segmentation_paths = self.worklist.iloc[:, 1]
            missing = segmentation_paths.isna() | segmentation_paths.astype(str).str.strip().eq("")
            if not self.allow_missing_segmentations and missing.any():
                # CSV row numbers include the header and are independent of
                # any custom DataFrame index supplied by a caller.
                rows = (np.flatnonzero(missing.to_numpy()) + 2).tolist()
                raise ValueError(
                    "Every volume needs a corresponding segmentation path when "
                    f"use_segmentations=True; missing paths on CSV rows {rows[:10]}"
                )

        if dataset_class is not None:
            self.dataset = dataset_class(file_obj=self.file_obj, **dataset_config)
            selected_collate_fn = collate_fn
        elif self.use_segmentations:
            # The generic loader already supports missing or unreadable masks;
            # set this explicitly so allow_missing_segmentations controls both blank
            # paths and paths that do not exist (or cannot be aligned).
            dataset_config["allow_missing_segmentations"] = self.allow_missing_segmentations
            self.dataset = FeatureExtractionDataset(file_obj=self.file_obj, **dataset_config)
            selected_collate_fn = self._feature_collate
        else:
            dataset_config.setdefault("segmentation_column", None)
            self.dataset = GenericRadiologyDataset(
                file_obj=self.file_obj, 
                **dataset_config
            )
            selected_collate_fn = self._generic_collate
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=selected_collate_fn,
        )

    @staticmethod
    def _generic_collate(batch):
        # Generic samples may contain optional segmentation arrays and arbitrary
        # metadata, neither of which PyTorch's default collator handles reliably.
        return {
            "pid": [sample["pid"] for sample in batch],
            "sid": [sample["sid"] for sample in batch],
            "barcode": [sample["barcode"] for sample in batch],
            "image": torch.stack([torch.as_tensor(sample["image"]) for sample in batch]),
            "seg": [sample.get("seg") for sample in batch],
            "metadata": [sample.get("metadata") for sample in batch],
        }

    @staticmethod
    def _feature_collate(batch):
        images = torch.stack([sample["image"] for sample in batch])
        return {
            "pid": [sample["pid"] for sample in batch],
            "sid": [sample["sid"] for sample in batch],
            "barcode": [sample["barcode"] for sample in batch],
            "image": images,
            "metadata": [sample.get("metadata") for sample in batch],
        }

    def __len__(self):
        return len(self.dataset)

    def capture_slice(self, image, img_id):
        image = image.squeeze()
        middle = image[int(image.shape[0]/2), :, :].unsqueeze(-1).detach().cpu().numpy()
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
        Extract features from axial 2D slices. Dataset samples may have shape
        [D, H, W] or [C, D, H, W] (with batch dimensions added by the loader).
        If ``processor`` is None, each slice is passed directly as
        ``pixel_values`` with shape [1, C, H, W].
        The extracted features are saved to disk in either HDF5 or PyTorch format.
        """
        
        with torch.no_grad():
            loop = tqdm(self.dataloader)
            feature_dir = os.path.join(self.save_dir, f"axial_features_{fmname}")
            os.makedirs(feature_dir, exist_ok=True)
            broken_volumes = []

            for batch in loop:
                images = batch["image"]

                for i, volume in enumerate(images):
                    featureBlock = []
                    pid = batch['pid'][i]
                    sid = batch['sid'][i]
                    barcode = batch['barcode'][i]
                    if "metadata" in batch:
                        metadata = batch['metadata'][i]
                    if volume.ndim not in (3, 4) or any(s == 0 for s in volume.shape[-3:]):
                        loop.set_postfix_str(f'Warning: Volume has shape {volume.shape} for {pid} series {sid}. Skipping...')
                        broken_volumes.append((pid, sid))
                        continue
                    # A generic sample is [D,H,W]; custom datasets may add C.
                    has_channel_dim = volume.ndim == 4
                    slices = volume.shape[1] if has_channel_dim else volume.shape[0]

                    save_path = os.path.join(feature_dir, pid, f"{barcode}.{saveas}")
                    if os.path.exists(save_path):
                        loop.set_postfix_str(f'Features already extracted for {pid} series {sid}: {barcode}. Skipping...')
                        continue
                    for s in range(slices):
                        if has_channel_dim:
                            image_slice = volume[:, s, :, :]
                        else:
                            image_slice = volume[s, :, :].unsqueeze(0)

                        if processor is None:
                            model_input = {"pixel_values": image_slice.unsqueeze(0).to(device)}
                        else:
                            model_input = processor(image_slice.squeeze()).to(device)

                        features = model(**model_input)
                        pooled = (
                            features.get("pooler_output")
                            if isinstance(features, dict)
                            else getattr(features, "pooler_output", None)
                        )
                        if pooled is None:
                            raise ValueError(
                                f"Model {fmname!r} did not return pooler_output for an axial slice"
                            )
                        featureBlock.append(pooled)

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
        Extract features from 3D volumes. The input volume is expected to have shape [B, D, H, W],
        where B is the batch size, D is the depth (number of slices), and H and W are the height and width of each slice. 
        The extracted features are saved to disk in either HDF5 or PyTorch format.
        """

        with torch.no_grad():
            loop = tqdm(self.dataloader)
            feature_dir = os.path.join(self.save_dir, f"3d_features_{fmname}")
            os.makedirs(feature_dir, exist_ok=True)
            broken_volumes = []

            for batch in loop:
                images = batch["image"]

                for i, volume in enumerate(images):
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

    def run_all(self, device, saveas: str = "h5"):
        """Load each configured encoder through the model_zoo adapter registry."""
        for model_name in self.fms:
            config_key = "renalclip" if model_name == "renal-clip" else model_name
            model_config = self.model_configs.get(model_name, self.model_configs.get(config_key, {}))
            model, processor, mode = load_foundation_model(
                model_name,
                config=model_config,
                device=device,
            )

            if mode == "axial":
                self.run_axial_feature_extraction_default(
                    model, model_name, processor, device, saveas=saveas
                )
            elif mode == "3d":
                self.run_3d_feature_extraction_default(
                    model, model_name, processor, device, saveas=saveas
                )
            else:
                raise ValueError(
                    f"Model adapter {model_name!r} returned unsupported mode {mode!r}"
                )

            print(
                f"Finished feature extraction for {model_name}. "
                f"Features saved in {saveas} format in {self.save_dir}."
            )
            self.save_thumbnails = False
