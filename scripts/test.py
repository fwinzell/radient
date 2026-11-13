from torch.utils.data import DataLoader
import pandas as pd
import os
from pathlib import Path


import sys
import importlib.util

from radient.utils import match_segmentations
from radient.dataset.features import FeatureExtractionDataset, KidneyFeatureDataset, kidney_collate_fn
from radient.dataset.generic import GenericRadiologyDataset
from radient.dataset.viewer import VolumeSliceViewer

def get_worklist(worklist, ct_source, segmentation_dir):
    worklist = pd.read_csv(worklist, header=None, skiprows=1)

    metadata = pd.read_csv(os.path.join(ct_source, "metadata.csv"))
    
    if not Path(segmentation_dir).is_absolute():
        segmentation_dir = os.path.join(ct_source, segmentation_dir)

    file_obj = match_segmentations(
        worklist[0].tolist(), metadata, segmentation_dir
    )

    return file_obj

def test_feature_dataset(df):
    dataset = FeatureExtractionDataset(
        file_obj=df,
        orientation="RAS",
        crop_size_mm=(140, 140, 160),
        target_size=(128, 128, 32),
        lesion_label=2,
        fallback_mode="center_crop",
        window_level=50,
        window_width=500,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    batch = next(iter(loader))

    sliceViewer = VolumeSliceViewer(
            image=batch['image'][0]
        )
    sliceViewer.show()

def test_kidney_dataset(df, n=1):
    dataset = KidneyFeatureDataset(
        file_obj=df,
        orientation="RAS",
        crop_size_mm=(140, 140, 160),
        target_size=(32, 128, 128),
        kidney_label=1,
        lesion_label=2,
        window_level=50,
        window_width=500,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=kidney_collate_fn
    )

    for i in range(n):
        batch = next(iter(loader))

        if batch['left_present'][0]:
            sliceViewer = VolumeSliceViewer(
                    image=batch['left'][0]
                )
            sliceViewer.show()

        if batch['right_present'][0]:
                sliceViewer = VolumeSliceViewer(
                        image=batch['right'][0]
                    )
                sliceViewer.show()
    

def test_generic(df, n=1):
    dataset = GenericTCIADataset(
        file_obj=df,
        orientation="RAS")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    for i in range(n):
        batch = next(iter(loader))

        sliceViewer = VolumeSliceViewer(
                image=batch['ct'][0],
                segmentation=batch['seg'][0]
            )
        sliceViewer.show()


def view_h5_file(path):
    import h5py
    with h5py.File(path, "r") as f:
        features_group = f['features']
        metadata = dict(features_group.attrs)

        def explore(name, obj):
            attrs = dict(obj.attrs)
            kind = "Dataset" if isinstance(obj, h5py.Dataset) else "Group"
            print(f"{name} ({kind}): attrs={attrs}")
        
        # check root attrs too
        print(f"/ (root): attrs={dict(f.attrs)}")
        f.visititems(explore)


if __name__ == "__main__":
    df = get_worklist(
        worklist="/data/pathology/projects/filip/cohorts/tcga/csv_files/radiology/tcga_kirc.csv",
    #    worklist="/Users/filipwinzell/Python/Radboud/cohorts/worklist.csv",
        ct_source="/Volumes/oncology/experiments/sarah_kidney_abnormality_segmentation/datasets/TCGA-KIRP/",
        segmentation_dir="/Volumes/oncology/experiments/sarah_kidney_abnormality_segmentation/datasets/TCGA-KIRC/onco_segmentations"
    )
    #test_generic(df, n=10)
    #test_feature_dataset(df)
    #test_kidney_dataset(df, n=3)
    view_h5_file("/Volumes/temporary/filip/cohorts/tcga/features/radiology/tcga_kirp_test/3d_features_renalclip/TCGA-B9-4114/f809ca11-ceb2-4ef4-82f5-424224296df0.h5")