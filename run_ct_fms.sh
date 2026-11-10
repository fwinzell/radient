#!/bin/bash
#SBATCH --qos=low
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=25G
#SBATCH --time=2-00:00:00
#SBATCH --no-container-entrypoint
#SBATCH --output=/home/%u/logs/slurm-%j.out \
#SBATCH --container-mounts=/data/pathology/projects/:/data/pathology/projects/,/data/pa_cpgarchive/:/data/pa_cpgarchive/,/data/oncology/:/data/oncology/
#SBATCH --container-image="dockerdex.umcn.nl:5005#fwinzell/ct_fms:latest"

pip3 install tensorboard

PROJECT_USER="filip"
PROJECT_ROOT="/data/pathology/projects/filip/code/"


SCRIPT_PATH="$PROJECT_ROOT"/multimodal_framework/scripts/feature_encoders/radiology/feature_extraction.py
IMG_DIR="/data/oncology/experiments/sarah_kidney_abnormality_segmentation/datasets/TCGA-KIRC/"
COHORT="kirc"
# Optional segmentations folder
# If you do not want to use segmentations, remember to remove the --use_segmentations flag from the python call below
SEGMENTATIONS="/data/oncology/experiments/sarah_kidney_abnormality_segmentation/datasets/TCGA-KIRC/onco_segmentations"

WORKLIST_PATH="/data/pathology/projects/filip/cohorts/tcga/csv_files/radiology/tcga_${COHORT}_worklist.csv"
SAVE_DIR="/data/pathology/projects/filip/cohorts/tcga/features/radiology/tcga_$COHORT"

if [ ! -f "$WORKLIST_PATH" ]; then
    echo "Could not find worklist, generating new ..."
    python3 "$PROJECT_ROOT/multimodal_framework/scripts/feature_encoders/radiology/create_worklist.py" \
        "$IMG_DIR" \
        "$WORKLIST_PATH" \
        --ext ".mha" \
        --keywords_avoid "AI segmentation" 
fi

mkdir -p "$HOME/logs"


# -------- Setup cache strategy: temp download + persistent storage --------

PERSISTENT_CACHE_DIR="/data/pathology/projects/${PROJECT_USER}/.cache"

echo "Creating temporary cache directory inside container..."
TEMP_CACHE_DIR=$(python3 -c "import tempfile; print(tempfile.mkdtemp(prefix='trident_cache_'))")

echo "Cache directories:"
echo "  Temporary cache (container): $TEMP_CACHE_DIR"
echo "  Persistent cache (storage): $PERSISTENT_CACHE_DIR"

# Set environment variables to use temp cache
export HF_HOME="$TEMP_CACHE_DIR/huggingface"
export TORCH_HOME="$TEMP_CACHE_DIR/torch"
export TRANSFORMERS_CACHE="$TEMP_CACHE_DIR/huggingface"
export TRIDENT_CACHE_DIR="$TEMP_CACHE_DIR/ct_fms"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TRIDENT_CACHE_DIR"

# Create persistent cache directories and copy existing cache if available
mkdir -p "$PERSISTENT_CACHE_DIR/huggingface" "$PERSISTENT_CACHE_DIR/torch" "$PERSISTENT_CACHE_DIR/ct_fms" 2>/dev/null || true

echo "Checking for existing cache in persistent storage..."
if [ -d "$PERSISTENT_CACHE_DIR/huggingface" ] && [ "$(ls -A $PERSISTENT_CACHE_DIR/huggingface 2>/dev/null)" ]; then
    echo "  Found existing HuggingFace cache, copying to container temp location..."
    cp -r "$PERSISTENT_CACHE_DIR/huggingface"/* "$HF_HOME/" 2>/dev/null || true
fi

if [ -d "$PERSISTENT_CACHE_DIR/torch" ] && [ "$(ls -A $PERSISTENT_CACHE_DIR/torch 2>/dev/null)" ]; then
    echo "  Found existing PyTorch cache, copying to container temp location..."
    cp -r "$PERSISTENT_CACHE_DIR/torch"/* "$TORCH_HOME/" 2>/dev/null || true
fi

# Currently supports: curia, tap-ct-2d, tap-ct-3d, merlin and renalclip
python3 "$SCRIPT_PATH" \
    --img_dir "$IMG_DIR" \
    --use_segmentations \
    --segmentation_dir "$SEGMENTATIONS" \
    --save_dir "$SAVE_DIR" \
    --worklist_csv "$WORKLIST_PATH" \
    --fm_list curia tap-ct-3d renalclip merlin \
    --save_type h5 \
    --hf_token_path "$PROJECT_ROOT/hf_token.txt" \
    --renalclip_repo_path "$PROJECT_ROOT/RenalCLIP/" \
    --organ "kidney" \
    --dataset_config_file "$PROJECT_ROOT/multimodal_framework/scripts/feature_encoders/radiology/config/tcga_kidney.yaml"
