#done by Sebastian Bastida Marin
#This script is for sbatch jobs on hpc3
#This runs the sql confidence job and writes outputs

# Contents:
#config vars set run options
#auto probe block picks a probe file when requested
#env setup loads python and paths
#run block executes sql.py

#SBATCH --job-name=tabletalk_sql
#SBATCH --account=cs175a_class_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:2
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --output=outputs/sql_%j.log
#SBATCH --error=outputs/sql_%j.err

#config vars
K=${K:-3}
LIMIT=${LIMIT:-0}
BIRD_DEV=${BIRD_DEV:-/data/class/cs175a/public/BIRD/dev_20240627/dev.json}
BASE_MODEL=${BASE_MODEL:-qwen}
FIXER_MODEL=${FIXER_MODEL:-arctic}
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_P=${TOP_P:-0.95}
MAX_TOKENS=${MAX_TOKENS:-512}
PROBE_PATH=${PROBE_PATH:-auto}

#strict shell mode
set -euo pipefail

#auto select latest probe
if [ "$PROBE_PATH" = "auto" ]; then
    LATEST_PROBE=$(ls -t outputs/probe_trained_*.pt 2>/dev/null | head -n 1 || true)
    if [ -n "$LATEST_PROBE" ]; then
        PROBE_PATH="$LATEST_PROBE"
        echo "Auto-selected latest trained probe: $PROBE_PATH"
    else
        echo "WARNING: PROBE_PATH=auto but no outputs/probe_trained_*.pt found. Running without probe."
        PROBE_PATH=""
    fi
fi

#go to repo
REPO_DIR="$HOME/TableTalk"
cd "$REPO_DIR"

#load python env
module load python/3.10.2
source .venv/bin/activate

#install runtime deps
pip install -q -U "transformers>=4.50.0" "bitsandbytes>=0.46.1" accelerate

#set bird path
export BIRD_PATH=/data/class/cs175a/public/BIRD

#set db dir
DEV_DB_DIR=${DEV_DB_DIR:-/data/class/cs175a/public/BIRD/dev_20240627/dev_databases}
export DEV_DB_DIR

#set model cache
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

#make output dir
mkdir -p outputs

#job header
echo "TableTalk SQL — Combined Confidence"
echo "dataset: bird_json"
echo "path: ${BIRD_DEV}"
echo "limit: ${LIMIT} k=${K}"
echo "base: ${BASE_MODEL} fixer: ${FIXER_MODEL}"
echo "probe: ${PROBE_PATH:-<none>}"
echo "db_dir: ${DEV_DB_DIR}"
echo "job id: ${SLURM_JOB_ID}"
echo "node: $(hostname)"

#set pytorch alloc behavior
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#optional probe arg
PROBE_ARG=()
[ -n "$PROBE_PATH" ] && PROBE_ARG=(--probe_path "$PROBE_PATH")

#run sql job
python scripts/sql.py \
    --dataset bird_json \
    --path "${BIRD_DEV}" \
    --k "${K}" \
    --limit "${LIMIT}" \
    --base_model "${BASE_MODEL}" \
    --fixer_model "${FIXER_MODEL}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --max_tokens "${MAX_TOKENS}" \
    "${PROBE_ARG[@]}"

echo "Done!! outputs written to outputs/ folder"
