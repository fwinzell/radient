from radient.RadiologyProcessor import RadiologyProcessor

from typing import List, Optional

import argparse
import torch
import numpy as np

from pydantic import BaseModel, ConfigDict
import yaml

class DatasetConfig(BaseModel):

    orientation: str = "LPS"
    spatial_size: int = 512

    rel_pid_idx: int = 0
    segmentation_labels: List[int] = [1,2]

    batch_size: int = 1

    crop_size_mm: tuple[float, float, float] = (140.0, 140.0,160.0)
    target_size: tuple[int, int, int] = (32,128,128)
    
    pad_value: float = -1000.0
    
    # HU normalization
    window_level: Optional[float] = 50.0
    window_width: Optional[float] = 500.0
    
    # Segmentation labels
    kidney_label: int = 1
    lesion_label: int = 2
    
    # Minimum kidney component size in voxels
    min_kidney_voxels: int = 1000
    return_metadata: bool = True

    @classmethod
    def from_yaml(cls, path: str | None = None) -> "DatasetConfig":
        if path is None:
            return cls()
        with open(path, "r") as f:
            yaml_dict = yaml.safe_load(f) or {}
        return cls(**yaml_dict)
    

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run feature extraction from radiology images")
    parser.add_argument('--save_dir', type=str, help="Path to the directory where the results of extracted features will be saved e.g. /data/pathology/projects/...",
                        default="/data/pathology/projects/filip/TCGA-data/features/radiology/")
    parser.add_argument('--img_dir', type=str, help="Legacy image directory option; input paths are read from the worklist CSV",
                        default="/data/oncology/experiments/sarah_kidney_abnormality_segmentation/datasets/TCGA-KIRC/")
    parser.add_argument('--worklist_csv', type=str, help="CSV with one image path per row; with --use_segmentations, column 1 is the image path and column 2 is its matching segmentation path.",
                        default="/data/pathology/projects/filip/TCGA-data/radiology_features/csv_files/ct_worklist.csv")
    parser.add_argument('--use_segmentations', action='store_true', help="Use segmentation-aware preprocessing. The worklist must have volume paths in column 1 and matching segmentation paths in column 2.")
    parser.add_argument('--allow_missing_segmentations', dest='require_segmentations', action='store_false', default=True, help="Allow blank or unavailable segmentation paths; affected cases use the dataset's no-segmentation fallback.")
    parser.add_argument('--fm_list', type=str, nargs='+', default=['merlin'], help="List of names of foundation models, Currently supports curia, tap-ct-2d, tap-ct-3d, merlin and renalclip.")
    parser.add_argument('--save_type', type=str, default="h5", help="Which file type to save the features in")
    parser.add_argument('--hf_token_path', type=str, default=None, help="Path to HuggingFace token for authentication. If not provided, will look for HF_TOKEN environment variable.")
    parser.add_argument('--model_config_file', type=str, default=None, help="YAML file with model-specific options keyed by foundation model name.")

    parser.add_argument('--dataset_config_file', type=str, default=None, help='Path to .yaml file with dataset config. If none default values will be used')

    return parser

def get_config():
    return build_parser().parse_args()

def load_dataset_config(config):
    dcfg = DatasetConfig.from_yaml(config.dataset_config_file)
    return dcfg.model_dump()


def load_model_configs(path):
    if path is None:
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def login_to_huggingface(path_to_token: Optional[str] = None):
    from huggingface_hub import login
    import os

    if path_to_token is not None:
        with open(path_to_token, 'r') as f:
            hf_token = f.read().strip()
    else:
        hf_token = os.getenv("HF_TOKEN")

    if hf_token is None:
        raise ValueError("HF_TOKEN environment variable not set")
    
    login(token=hf_token)

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_config = load_dataset_config(config)

    processor = RadiologyProcessor(
        save_dir=config.save_dir,
        worklist=config.worklist_csv,
        fms=config.fm_list,
        image_extension=".mha",
        use_segmentations=config.use_segmentations,
        require_segmentations=config.require_segmentations,
        dataset_config=dataset_config,
        model_configs=load_model_configs(config.model_config_file),
        save_thumbnails=True,
    )

    processor.run_all(device, config.save_type)


if __name__ == "__main__":
    config = get_config()
    login_to_huggingface(config.hf_token_path)
    main(config)   
