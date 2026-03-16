#done by Logan Mifflin
#This script is for sbatch jobs on hpc3
#This collects whitebox probe training data and writes outputs

# Contents:
#config vars set train run options
#env setup loads python and paths
#run block starts train_probe.py

#SBATCH --job-name=tabletalk_train_probe
#SBATCH --account=cs175a_class_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:2
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --output=outputs/train_probe_%j.log
#SBATCH --error=outputs/train_probe_%j.err

#config vars
LIMIT=${LIMIT:-0}
BIRD_TRAIN=${BIRD_TRAIN:-}
BASE_MODEL=${BASE_MODEL:-qwen}
FIXER_MODEL=${FIXER_MODEL:-arctic}
MAX_TOKENS=${MAX_TOKENS:-2048}
PROJ_DIM=${PROJ_DIM:-256}
PCA_BATCH=${PCA_BATCH:-256}
SKIP=${SKIP:-0}
RESUME=${RESUME:-}

#strict shell mode
set -euo pipefail

#go to repo
REPO_DIR="$HOME/TableTalk"
cd "$REPO_DIR"

#load python env
module load python/3.10.2
source .venv/bin/activate

#install runtime deps
pip install -q -U "transformers>=4.50.0" "bitsandbytes>=0.46.1" accelerate scikit-learn

#set bird root
export BIRD_PATH=/data/class/cs175a/public/BIRD

#auto detect train json
if [ -z "${BIRD_TRAIN}" ]; then
    for cand in \
        "${BIRD_PATH}/train/train.json" \
        "${BIRD_PATH}/train.json" \
        "${BIRD_PATH}/train/train_set.json"
    do
        if [ -f "${cand}" ]; then
            BIRD_TRAIN="${cand}"
            break
        fi
    done
fi

#fail fast if train json is missing
if [ ! -f "${BIRD_TRAIN}" ]; then
    echo "ERROR: BIRD train file not found."
    echo "  BIRD_PATH=${BIRD_PATH}"
    echo "  BIRD_TRAIN=${BIRD_TRAIN:-<unset>}"
    echo "Set it explicitly, for example:"
    echo "  sbatch --export=ALL,BIRD_TRAIN=/path/to/train.json slurm/run_train_probe.sh"
    exit 1
fi

#set train db dir
TRAIN_DB_DIR=${TRAIN_DB_DIR:-/data/class/cs175a/public/BIRD/train/train_databases}
export DEV_DB_DIR="$TRAIN_DB_DIR"

#set model cache
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

#set pytorch alloc behavior
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#make output dir
mkdir -p outputs

#job header
echo "TableTalk — Whitebox Probe Training Data"
echo "dataset: bird_json (train split)"
echo "path: ${BIRD_TRAIN}"
echo "db_dir: ${TRAIN_DB_DIR}"
echo "limit: ${LIMIT} skip: ${SKIP}"
echo "base: ${BASE_MODEL} fixer: ${FIXER_MODEL}"
echo "proj_dim:${PROJ_DIM} pca_batch:${PCA_BATCH}"
echo "resume: ${RESUME:-<fresh run>}"
echo "job id: ${SLURM_JOB_ID}"
echo "node: $(hostname)"

#optional resume arg
RESUME_ARG=()
[ -n "${RESUME}" ] && RESUME_ARG=(--resume "${RESUME}")

#run train probe job
python scripts/train_probe.py \
    --dataset bird_json \
    --path "${BIRD_TRAIN}" \
    --limit "${LIMIT}" \
    --base_model "${BASE_MODEL}" \
    --fixer_model "${FIXER_MODEL}" \
    --max_tokens "${MAX_TOKENS}" \
    --proj_dim "${PROJ_DIM}" \
    --pca_batch "${PCA_BATCH}" \
    --skip "${SKIP}" \
    "${RESUME_ARG[@]}"

echo "Done!! outputs written to outputs/ folder"
