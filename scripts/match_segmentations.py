from pathlib import Path
import os
import pandas as pd
import numpy as np
import argparse


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


def update_worklist_with_segmentations(
        worklist_path: Path,
        metadata_path: Path,
        seg_dir: Path | str,
        filter_missing: bool = True,
    ):

    worklist = pd.read_csv(worklist_path, header=None, skiprows=1)[0].tolist()
    metadata = pd.read_csv(metadata_path)

    df = match_segmentations(worklist, metadata, seg_dir)
    df.columns = ["worklist", "segmentation"]

    save_path = worklist_path.parent / f"{worklist_path.stem}_with_segmentations.csv"
    df.to_csv(save_path, index=False)
    print(f"Saved updated worklist with segmentations to {save_path}")
    return df

def main():
    """
    Match segmentations to a worklist using the metadata file.

    Example usage:
        python match_segmentations.py \
            --worklist /path/to/worklist.csv \
            --metadata /path/to/metadata.csv \
            --seg_dir /path/to/segmentations/

    Real example:
        python3 scripts/match_segmentations.py \
            --worklist /data/pathology/projects/filip/cohorts/radient_worklists/tcga_kirc.csv \
            --metadata /data/pa_cpgarchive/projects/multimodal_baseline/kidney/datasets/TCGA-KIRC/radiology/metadata.csv \
            --seg_dir /data/pa_cpgarchive/projects/multimodal_baseline/kidney/datasets/TCGA-KIRC/radiology/onco_segmentations
    """
    parser = argparse.ArgumentParser(
        description="Match segmentations to a worklist using the metadata file."
    )
    parser.add_argument("--worklist", required=True, help="Path to the worklist CSV file")
    parser.add_argument("--metadata", required=True, help="Path to the metadata CSV file")
    parser.add_argument("--seg_dir", required=True, help="Path to the segmentation directory")
    args = parser.parse_args()

    update_worklist_with_segmentations(
        worklist_path=Path(args.worklist),
        metadata_path=Path(args.metadata),
        seg_dir=Path(args.seg_dir)
    )

if __name__ == "__main__":
    main()