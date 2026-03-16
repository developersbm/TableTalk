#done by Rei Shindo
#This script is for sbatch jobs on hpc3 for nonsql.py runs
#This runs code confidence jobs and writes outputs

# Contents:
#config vars set run options
#env setup loads python and venv
#run block executes humaneval and mbpp jobs

#SBATCH --job-name=tabletalk_code
#SBATCH --account=cs175a_class_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:2
#SBATCH --mem=230G
#SBATCH --time=48:00:00
#SBATCH --output=outputs/code_%j.log
#SBATCH --error=outputs/code_%j.err

#config vars
K=${K:-3}
LIMIT=${LIMIT:-0}
BASE_MODEL=${BASE_MODEL:-gemma}
FIXER_MODEL=${FIXER_MODEL:-qwen}
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_P=${TOP_P:-0.95}
MAX_TOKENS=${MAX_TOKENS:-512}

#strict shell mode
set -euo pipefail

#go to repo
REPO_DIR="$HOME/TableTalk"
cd "$REPO_DIR"

#load python env
module load python/3.10.2
source .venv/bin/activate

#install runtime deps
pip install -q -U "transformers>=4.50.0" "bitsandbytes>=0.46.1" accelerate

#set model cache
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

#make output dir
mkdir -p outputs

#job header
echo "TableTalk Code — Combined Confidence"
echo "datasets: humaneval + mbpp"
echo "limit: ${LIMIT}  k=${K}"
echo "base: ${BASE_MODEL}  fixer: ${FIXER_MODEL}"
echo "job id: ${SLURM_JOB_ID}"
echo "node: $(hostname)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#run humaneval job
echo "Running HumanEval+"
python scripts/nonsql.py \
    --dataset humaneval \
    --k "${K}" \
    --limit "${LIMIT}" \
    --base_model "${BASE_MODEL}" \
    --fixer_model "${FIXER_MODEL}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --max_tokens "${MAX_TOKENS}"

#run mbpp job
echo "Running MBPP+"
python scripts/nonsql.py \
    --dataset mbpp \
    --k "${K}" \
    --limit "${LIMIT}" \
    --base_model "${BASE_MODEL}" \
    --fixer_model "${FIXER_MODEL}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --max_tokens "${MAX_TOKENS}"

echo "Done!! outputs written to outputs/ folder"
