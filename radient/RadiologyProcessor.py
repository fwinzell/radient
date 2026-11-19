import os
import torch
from tqdm import tqdm
import pandas as pd
import SimpleITK as sitk
import numpy as np
from pathlib import Path
import cv2
import yaml

from typing import List, Any, Optional, Union, Callable, Type

from torch.utils.data import DataLoader

from radient.utils import save_h5
from radient.dataset.generic import GenericRadiologyDataset
from radient.dataset.features import (
    FeatureExtractionDataset,
    KidneyFeatureDataset,
    kidney_collate_fn,
)
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
        segmentation_mode: Optional[str] = None,
        totalsegmentator_rois: Optional[List[str]] = None,
        totalsegmentator_task: str = "total",
        totalsegmentator_fast: bool = False,
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

        if segmentation_mode is None:
            segmentation_mode = "external" if use_segmentations else "none"
        if segmentation_mode not in {"none", "external", "totalsegmentator"}:
            raise ValueError("segmentation_mode must be 'none', 'external', or 'totalsegmentator'")
        if use_segmentations and segmentation_mode != "external":
            raise ValueError("--use_segmentations can only be combined with segmentation_mode='external'")

        if segmentation_mode == "external" and not isinstance(worklist, pd.DataFrame):
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
        default_model_config_path = Path(__file__).resolve().parents[1] / "config" / "model_configs.yaml"
        self.model_configs = {}
        if default_model_config_path.exists():
            with default_model_config_path.open("r", encoding="utf-8") as config_file:
                self.model_configs = yaml.safe_load(config_file) or {}
        for name, settings in (model_configs or {}).items():
            model_settings = self.model_configs.setdefault(name, {})
            for key, value in settings.items():
                if isinstance(value, dict) and isinstance(model_settings.get(key), dict):
                    model_settings[key].update(value)
                else:
                    model_settings[key] = value
        self.save_thumbnails = save_thumbnails

        # Validate extension
        assert self.image_extension.startswith('.'), f'Invalid extension: {self.image_extension} (must start with a period)'

        self.file_obj = self.worklist
        self.segmentation_mode = segmentation_mode
        self.use_segmentations = segmentation_mode == "external"
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

        if dataset_class is None:
            if segmentation_mode == "external":
                dataset_config["allow_missing_segmentations"] = self.allow_missing_segmentations
            else:
                dataset_config["segmentation_column"] = None
            dataset_config["segmentation_mode"] = segmentation_mode
            if segmentation_mode == "totalsegmentator":
                dataset_config["totalsegmentator_rois"] = totalsegmentator_rois or ["kidney_left", "kidney_right"]
                dataset_config["totalsegmentator_task"] = totalsegmentator_task
                dataset_config["totalsegmentator_fast"] = totalsegmentator_fast
            if segmentation_mode == "none":
                dataset_config.setdefault("fallback_mode", "whole_volume")

        self.dataset_config = dataset_config
        self.dataset_class = dataset_class
        self.custom_collate_fn = collate_fn
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = None
        self.dataloader = None
        self._active_processor = None
        self._configure_dataset(preprocessor=None, modality=dataset_config.get("modality", "ct"))

    def _configure_dataset(
        self,
        preprocessor,
        modality,
        preprocessor_strategy="volume",
        dataset_overrides=None,
    ):
        """Build a fresh dataset/loader for the selected model."""
        modality = str(modality).lower()
        if modality not in {"ct", "mri"}:
            raise ValueError("modality must be 'ct' or 'mri'")

        dataset_options = {**self.dataset_config, **(dataset_overrides or {})}
        dataset_options.pop("modality", None)
        dataset_type = dataset_options.pop("dataset_type", "volume")
        kidney_dataset = dataset_type == "kidney" or self.dataset_class is KidneyFeatureDataset
        if self.dataset_class is not None and not kidney_dataset:
            self.dataset = self.dataset_class(
                file_obj=self.file_obj,
                **dataset_options,
            )
            collate_fn = self.custom_collate_fn
            self._active_processor = preprocessor
        elif kidney_dataset:
            self.dataset = KidneyFeatureDataset(
                file_obj=self.file_obj,
                modality=modality,
                preprocessor=preprocessor,
                preprocessor_strategy=preprocessor_strategy,
                **dataset_options,
            )
            collate_fn = kidney_collate_fn
            self._active_processor = None
        elif dataset_type == "volume":
            self.dataset = FeatureExtractionDataset(
                file_obj=self.file_obj,
                modality=modality,
                preprocessor=preprocessor,
                preprocessor_strategy=preprocessor_strategy,
                **dataset_options,
            )
            collate_fn = self._feature_collate
            self._active_processor = None
        else:
            raise ValueError("dataset_type must be 'volume' or 'kidney'")

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
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
        

    @staticmethod
    def _model_features(output):
        """Get the feature tensor from common Transformers output types."""
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, dict):
            for key in ("pooler_output", "last_hidden_state"):
                if output.get(key) is not None:
                    return output[key]
        for key in ("pooler_output", "last_hidden_state"):
            value = getattr(output, key, None)
            if value is not None:
                return value
        raise ValueError("Model output did not contain a feature tensor")

    @staticmethod
    def _volume_input(volume):
        """Convert [D,H,W] or [C,D,H,W] samples to [1,C,D,H,W]."""
        if volume.ndim == 3:
            volume = volume.unsqueeze(0).unsqueeze(0)
        elif volume.ndim == 4:
            volume = volume.unsqueeze(0)
        if volume.ndim != 5:
            raise ValueError(
                f"Expected a [D,H,W] or [C,D,H,W] sample, got {tuple(volume.shape)}"
            )
        return volume

    def _extract_axial_features(self, volume, model, processor, device):
        """Apply the processor and model to each axial slice in a volume."""
        volume = volume if volume.ndim == 4 else volume.unsqueeze(0)
        slice_features = []
        for image_slice in volume.permute(1, 0, 2, 3):
            model_input = (
                {"pixel_values": image_slice.unsqueeze(0).to(device)}
                if processor is None
                else processor(image_slice.squeeze()).to(device)
            )
            slice_features.append(self._model_features(model(**model_input)))
        return torch.cat(slice_features, dim=0)

    def _extract_volume_features(self, volume, model, processor, device, model_input_strategy):
        """Preprocess a full volume and run the configured model input layout."""
        model_input = self._volume_input(volume)
        if processor is not None:
            model_input = processor(model_input)["pixel_values"]

        if model_input_strategy == "slice_batch":
            # A 2D encoder can preprocess the full volume but still expects
            # [B*D,C,H,W] at forward time. The processor may pad D, so axial
            # saving later trims features back to the source depth.
            if model_input.ndim == 5:
                batch, channels, depth, height, width = model_input.shape
                model_input = model_input.permute(0, 2, 1, 3, 4).reshape(
                    batch * depth, channels, height, width
                )
            elif model_input.ndim != 4:
                raise ValueError(
                    "Slice batch input must be [B,C,D,H,W] or [N,C,H,W], "
                    f"got {tuple(model_input.shape)}"
                )
        elif model_input_strategy == "volume":
            if model_input.ndim == 4:
                model_input = model_input.unsqueeze(0)
            if model_input.ndim != 5:
                raise ValueError(
                    f"Volume input must be [B,C,D,H,W], got {tuple(model_input.shape)}"
                )
        else:
            raise ValueError(
                "model_input_strategy must be 'volume' or 'slice_batch', "
                f"got {model_input_strategy!r}"
            )
        return self._model_features(model(model_input.to(device)))

    @staticmethod
    def _axial_layout(features, depth):
        """Drop a singleton batch axis and verify one feature row per slice."""
        if features.ndim >= 3 and features.shape[0] == 1:
            features = features.squeeze(0)
        if features.ndim < 2 or features.shape[0] < depth:
            raise ValueError(
                "Axial saving requires one feature vector per input slice; "
                f"got feature shape {tuple(features.shape)} for depth {depth}."
            )
        return features[:depth]

    def _write_features(self, features, save_path, saveas, metadata, pid, sid, fmname, feature_dir):
        """Write feature tensors and metadata to HDF5 or PyTorch."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if saveas == "h5":
            save_h5(
                save_path=save_path,
                assets={"features": features.detach().cpu().numpy(), "metadata": metadata},
                attributes={
                    "features": {
                        "pid": pid,
                        "series": sid,
                        "savetodir": feature_dir,
                        "encoder": fmname,
                    }
                },
                mode="w",
            )
        elif saveas == "pt":
            torch.save(features.detach().cpu(), save_path)
        else:
            raise ValueError(f"Invalid save format {saveas!r}; expected 'h5' or 'pt'.")

    def _save_axial_features(self, features, depth, save_path, saveas, metadata, pid, sid, fmname, feature_dir):
        features = self._axial_layout(features, depth)
        self._write_features(features, save_path, saveas, metadata, pid, sid, fmname, feature_dir)

    def _save_volume_features(self, features, depth, save_path, saveas, metadata, pid, sid, fmname, feature_dir):
        self._write_features(features, save_path, saveas, metadata, pid, sid, fmname, feature_dir)

    def _run_kidney_extraction(
        self,
        model,
        fmname,
        processor,
        device,
        saveas,
        extraction_strategy,
        save_strategy,
        model_input_strategy,
        feature_dir,
    ):
        saver = self._save_axial_features if save_strategy == "axial" else self._save_volume_features
        with torch.no_grad():
            loop = tqdm(self.dataloader)
            for batch in loop:
                for index, pid in enumerate(batch["pid"]):
                    sid = batch["sid"][index]
                    barcode = batch["barcode"][index]
                    metadata = batch.get("metadata", [None] * len(batch["pid"]))[index]

                    for side in ("left", "right"):
                        volume = batch[side][index]
                        if volume is None:
                            continue
                        if volume.ndim not in (3, 4) or any(size == 0 for size in volume.shape[-3:]):
                            loop.set_postfix_str(
                                f"Skipping invalid {side} kidney crop {tuple(volume.shape)} for {pid}"
                            )
                            continue

                        save_path = os.path.join(
                            feature_dir, str(pid), f"{barcode}_{side}.{saveas}"
                        )
                        if os.path.exists(save_path):
                            loop.set_postfix_str(
                                f"Features already extracted for {pid} {side} kidney"
                            )
                            continue

                        side_metadata = dict(metadata or {})
                        side_metadata["kidney_side"] = side
                        preprocessing = side_metadata.get(f"{side}_preprocessing") or {}
                        input_shape = preprocessing.get("input_shape_zyx")
                        depth = input_shape[0] if input_shape else volume.shape[-3]
                        if extraction_strategy == "axial":
                            features = self._extract_axial_features(
                                volume, model, processor, device
                            )
                        else:
                            features = self._extract_volume_features(
                                volume, model, processor, device, model_input_strategy
                            )
                        if self.save_thumbnails:
                            self.capture_slice(volume, f"{pid}_{barcode}_{side}")
                        saver(
                            features, depth, save_path, saveas, side_metadata,
                            pid, sid, fmname, feature_dir,
                        )
                        loop.set_postfix_str(
                            f"Saved {side} kidney features for {pid} series {sid}"
                        )

    def _run_extraction(
        self,
        model,
        fmname,
        processor,
        device,
        saveas,
        extraction_strategy,
        save_strategy,
        model_input_strategy="volume",
    ):
        if extraction_strategy not in {"axial", "volume"}:
            raise ValueError("extraction_strategy must be 'axial' or 'volume'")
        if save_strategy not in {"axial", "volume"}:
            raise ValueError("save_strategy must be 'axial' or 'volume'")
        if extraction_strategy == "axial" and save_strategy == "volume":
            raise ValueError("Axial extraction cannot produce volume-level features")

        feature_dir = os.path.join(
            self.save_dir,
            f"{'axial' if save_strategy == 'axial' else '3d'}_features_{fmname}",
        )
        os.makedirs(feature_dir, exist_ok=True)
        if isinstance(self.dataset, KidneyFeatureDataset):
            self._run_kidney_extraction(
                model,
                fmname,
                processor,
                device,
                saveas,
                extraction_strategy,
                save_strategy,
                model_input_strategy,
                feature_dir,
            )
            return
        broken_volumes = []

        with torch.no_grad():
            loop = tqdm(self.dataloader)
            for batch in loop:
                for i, volume in enumerate(batch["image"]):
                    pid = batch["pid"][i]
                    sid = batch["sid"][i]
                    barcode = batch["barcode"][i]
                    metadata = batch.get("metadata", [None] * len(batch["image"]))[i]
                    if volume.ndim not in (3, 4) or any(size == 0 for size in volume.shape[-3:]):
                        loop.set_postfix_str(
                            f"Skipping invalid volume {tuple(volume.shape)} for {pid} series {sid}"
                        )
                        broken_volumes.append((pid, sid))
                        continue

                    save_path = os.path.join(feature_dir, str(pid), f"{barcode}.{saveas}")
                    if os.path.exists(save_path):
                        loop.set_postfix_str(f"Features already extracted for {pid} series {sid}: {barcode}")
                        continue

                    processor_input_shape = (
                        metadata.get("processor_input_shape")
                        if isinstance(metadata, dict)
                        else None
                    )
                    depth = (
                        processor_input_shape[-3]
                        if processor_input_shape is not None
                        else volume.shape[-3]
                    )
                    if extraction_strategy == "axial":
                        features = self._extract_axial_features(volume, model, processor, device)
                    else:
                        features = self._extract_volume_features(
                            volume, model, processor, device, model_input_strategy
                        )

                    if self.save_thumbnails:
                        self.capture_slice(volume, f"{pid}_{barcode}")

                    saver = self._save_axial_features if save_strategy == "axial" else self._save_volume_features
                    saver(features, depth, save_path, saveas, metadata, pid, sid, fmname, feature_dir)
                    loop.set_postfix_str(f"Features saved for {pid} series {sid}: {barcode}.")

        if broken_volumes:
            print(f"Warning: skipped {len(broken_volumes)} volumes with invalid shapes:")
            for pid, sid in broken_volumes:
                print(f" - Patient ID: {pid}, Series ID: {sid}")

    def run_axial_feature_extraction_default(self, model, fmname, processor, device, saveas="h5", save_strategy="axial"):
        self._run_extraction(model, fmname, processor, device, saveas, "axial", save_strategy)

    def run_3d_feature_extraction_default(self, model, fmname, processor, device, saveas="h5", save_strategy="volume"):
        self._run_extraction(model, fmname, processor, device, saveas, "volume", save_strategy)

    def run_all(self, device, saveas: str = "h5"):
        """Load each configured encoder through the model_zoo adapter registry."""
        for model_name in self.fms:
            config_key = "renalclip" if model_name == "renal-clip" else model_name
            model_config = self.model_configs.get(model_name, self.model_configs.get(config_key, {}))
            dataset_overrides = model_config.get("dataset", {})
            modality = model_config.get(
                "modality",
                dataset_overrides.get("modality", self.dataset_config.get("modality", "ct")),
            )
            model_config = {**model_config, "modality": modality}
            extraction_strategy = model_config.get("extraction_strategy")
            model, processor = load_foundation_model(
                model_name,
                config=model_config,
                device=device,
            )
            self._configure_dataset(
                preprocessor=processor,
                modality=modality,
                preprocessor_strategy=extraction_strategy or "volume",
                dataset_overrides=dataset_overrides,
            )
            active_processor = self._active_processor

            if extraction_strategy is None:
                raise ValueError(
                    f"Model {model_name!r} has no extraction_strategy in model_configs.yaml"
                )
            save_strategy = model_config.get("save_strategy", extraction_strategy)
            model_input_strategy = model_config.get("model_input_strategy", "volume")
            self._run_extraction(
                model,
                model_name,
                active_processor,
                device,
                saveas,
                extraction_strategy,
                save_strategy,
                model_input_strategy,
            )

            print(
                f"Finished feature extraction for {model_name}. "
                f"Features saved in {saveas} format in {self.save_dir}."
            )
            self.save_thumbnails = False
