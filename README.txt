# TableTalk
Description: TableTalk studies how well LLMs express uncertainty in the code and SQL they generate. It evaluates different confidence methods and measures how accurately those confidence scores reflect the model’s true success rate.

### Mention the external libraries your team used (e.g., numpy, sklearn, etc)
Libraries used:
 - Hugging Face Transformers (https://huggingface.co/docs/transformers/)
 - PyTorch (https://pytorch.org/)
 - NumPy (https://numpy.org/)
 - Scikit-Learn (https://scikit-learn.org/)
 - SciPy (https://scipy.org/)
 - Matplotlib (https://matplotlib.org/)

### Publicly available codes used:
 - BIRD-SQL
    - (https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird). 
    - Used primarily without any modifications for the SQL evaluation.
 - EvalPlus
    - (https://github.com/evalplus/evalplus). 
    - Used without any modifications for humaneval and mbpp testing.

## Scripts/functions written by our team (and how many lines per file):
- src/verbalized_confidence.py -  Calculates verbalized confidence measures (580 lines)
- src/consistency_scoring.py -  Evaluates and scores self-consistency pipelines (310 lines)
- src/loader.py -  Loads datasets and structures the language models (360 lines)
- src/fixer.py -  Standardizes format and fixes pipeline anomalies (350 lines)
- src/llm_client.py -  Manages the LLM endpoints and prompt handling via HuggingFace (380 lines)
- src/whitebox_probing.py -  Creates PyTorch linear probes to extract internal embeddings (220 lines)
- src/tokenize_confidence.py -  Measures token-level probability confidences (110 lines)
- src/model_tokenizing.py -  Translates formatted text into token IDs efficiently (80 lines)
- src/diff_analysis.py -  Analyzes formatting differences in textual generation outputs (160 lines)
- src/inclusion.py - Handles straightforward substring definitions and validations (50 lines)
- src/scripts/train_probe.py -  Orchestrates the probing model's training pipeline (560 lines)
- src/scripts/fit_probe.py -  Evaluates and fits linear probing classification methods (160 lines)
- src/scripts/sql.py -  Execution script evaluating the SQL dataset tasks (510 lines)
- src/scripts/nonsql.py -  Execution script running non-SQL (like Python) tasks (440 lines)
- src/stats/plot_bss_ece.py -  Generates expected calibration error and Brier statistics charts (850 lines)
- src/stats/plot_bss_ece_sql.py -  Generates BSS and ECE visualizations restricted to SQL items (220 lines)
- src/stats/stats.py -  Generates global task outcome summary counts (120 lines)
- src/stats/sqlStats.py -  Computes granular database pass/fail performance stats (120 lines)
- src/slurm/run_code.sh - Slurm batch script to configure and submit non-SQL evaluation jobs (85 lines)
- src/slurm/run_fit_probe.sh - Slurm batch script to deploy the probe fitting workflows (84 lines)
- src/slurm/run_sql.sh - Slurm batch script to execute heavy SQL-based jobs across cluster nodes (104 lines)
- src/slurm/run_train_probe.sh - Slurm batch script to manage parallel tracking for probe training tasks (116 lines)
- requirements.txt - Lists all Python package dependencies required to run the project (9 lines)
- pyrightconfig.json - Configures the Pyright static type checker for the workspace (17 lines)
- .gitignore - Specifies intentionally untracked files to ignore in Git version control (27 lines)