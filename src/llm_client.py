#done by Rei Shindo and Sebastian Bastida Marin

#This file helps load large language models from huggingface and run text generation
#it manages gpu memory and records helpful resource logs while models load

# Content:
#_loadlogger: simple tool to track model loading time and log it
#_gpu_ram_summary: checks how much ram and gpu memory is currently used
#_auto_max_memory: figures out how much memory can safely be used on each gpu
#_basehfclient: main model client that handles tokenizers and generation
#_build_input: formats input text into model tokens
#generate_reference: makes a straightforward text generation and gives a score
#generate_with_full_logprobs: gets generation output with detailed token scores
#generate_samples: generates multiple randomized answers for the same prompt

import os
import threading
import torch
import time
from datetime import datetime
from typing import Any, List, Tuple, cast
class _LoadLogger:

    #how often to check memory in seconds
    HEARTBEAT_INTERVAL = 15

    #setup logging to output file based on job id
    def __init__(self, label: str = "model"):
        os.makedirs("outputs", exist_ok=True)
        job_id = os.getenv("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.path = os.path.join("outputs", f"llm_load_{job_id}.log")
        self._label = label
        self._start = time.monotonic()
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._fh = open(self.path, "a", buffering=1)
        self("log file: " + os.path.abspath(self.path))

    #start background logger
    def __enter__(self):
        self._thread.start()
        return self

    #stop background logger
    def __exit__(self, *_):
        self._stop_evt.set()
        self._thread.join(timeout=5)
        self._fh.close()

    #log a message with timestamp
    def __call__(self, msg: str):
        elapsed = time.monotonic() - self._start
        line = f"[{elapsed:7.1f}s] {msg}"
        print(f"[{self._label}] {line}", flush=True)
        self._fh.write(line + "\n")


    #log memory summary at intervals
    def _heartbeat(self):
        while not self._stop_evt.wait(timeout=self.HEARTBEAT_INTERVAL):
            self(_gpu_ram_summary())

    #check if model weights are saved locally
    @staticmethod
    def _cache_status(model_name: str) -> str:
        cache_dir = os.path.join(
            os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            "hub", "models--" + model_name.replace("/", "--"),
        )
        if os.path.isdir(cache_dir):
            try:
                import subprocess
                result = subprocess.run(
                    ["du", "-sh", cache_dir], capture_output=True, text=True, timeout=10
                )
                size = result.stdout.split()[0] if result.returncode == 0 else "?"
            except Exception:
                size = "?"
            return f"cache HIT — {cache_dir} ({size})"
        return f"cache MISS — will download to {cache_dir}"

#get current gpu and system memory usage
def _gpu_ram_summary() -> str:
    parts = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                alloc = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                parts.append(f"GPU{i}: {alloc:.1f}/{total:.0f}GB alloc ({reserved:.1f}GB res)")
    except Exception:
        pass
    try:
        import resource
        ram_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        parts.append(f"RAM_RSS: {ram_gb:.1f}GB")
    except Exception:
        pass
    return "  |  ".join(parts) if parts else "stats unavailable"

#set limits for safe gpu memory allocation
def _auto_max_memory() -> "dict[int | str, str]":
    n_gpus = torch.cuda.device_count()
    max_memory: dict[int | str, str] = {}
    for i in range(n_gpus):
        total_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
        max_memory[i] = f"{int(total_gb * 0.85)}GB"
    if not max_memory:
        max_memory["cpu"] = "20GB"
    return max_memory
#main class to talk to huggingface models
class _BaseHFClient:
    DEFAULT_MODEL: str = ""
    LOG_LABEL: str = "HF"
    GPU_MEMORY_GB: int = 0

    #initialize model and set up tokenizer
    def __init__(self, model_name: str | None = None, gpu_memory_gb: int | None = None):
        model_name = model_name or self.DEFAULT_MODEL
        gpu_budget = gpu_memory_gb or self.GPU_MEMORY_GB
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "transformers, torch, bitsandbytes, and accelerate are required.\n"
                "Run:  pip install transformers accelerate torch bitsandbytes"
            )

        offload_dir = os.path.join(os.getenv("TMPDIR", "/tmp"), "hf_offload", self.LOG_LABEL)
        os.makedirs(offload_dir, exist_ok=True)

        with _LoadLogger(self.LOG_LABEL) as log:
            log(f"loader start  model={model_name}")
            log(f"node: {os.uname().nodename}  "
                f"job: {os.getenv('SLURM_JOB_ID','local')}  "
                f"gpus: {os.getenv('SLURM_GPUS_ON_NODE', os.getenv('CUDA_VISIBLE_DEVICES','?'))}")
            log(_LoadLogger._cache_status(model_name))
            log(_gpu_ram_summary())

            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

            max_memory: dict[int | str, str]
            if gpu_budget > 0:
                n_gpus = torch.cuda.device_count()
                max_memory = {i: f"{gpu_budget}GB" for i in range(n_gpus)}
                max_memory["cpu"] = "80GB"
            else:
                max_memory = _auto_max_memory()
            log(f"max_memory: {max_memory}")

            log("loading tokenizer…")
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            log("tokenizer loaded")

            log("loading model weights (4-bit NF4)")

            import concurrent.futures as _cf
            _OrigTPE = _cf.ThreadPoolExecutor

            class _SeqExecutor(_OrigTPE):
                def __init__(self, *a, **kw):
                    kw["max_workers"] = 1
                    super().__init__(*a, **kw)

            try:
                import importlib

                _cml = importlib.import_module("transformers.core_model_loading")
                if hasattr(_cml, "ThreadPoolExecutor"):
                    setattr(_cml, "ThreadPoolExecutor", _SeqExecutor)
            except Exception:
                pass

            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map="auto",
                max_memory=max_memory,
                quantization_config=bnb_cfg,
                low_cpu_mem_usage=True,
                offload_folder=offload_dir,
            )
            self._model.eval()

            if hasattr(self._model, "generation_config"):
                self._model.generation_config.top_k = None
                self._model.generation_config.top_p = None

            log(_gpu_ram_summary())

        self._torch = torch
        self.model_name = model_name


    MAX_INPUT_TOKENS = 8192

    #turn text string into device tokens
    def _build_input(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt
        enc = self._tokenizer(text, return_tensors="pt")
        if enc["input_ids"].shape[1] > self.MAX_INPUT_TOKENS:
            print(f"prompt truncated from {enc['input_ids'].shape[1]} "
                  f"to {self.MAX_INPUT_TOKENS} tokens")
            enc["input_ids"] = enc["input_ids"][:, :self.MAX_INPUT_TOKENS]
            enc["attention_mask"] = enc["attention_mask"][:, :self.MAX_INPUT_TOKENS]
        return enc.to(self._model.device)

    #generate a single response and its score
    def generate_reference(
        self,
        prompt: str,
        max_tokens: int = 512,
    ) -> Tuple[str, float]:
        text, mean_lp, _ = self.generate_with_full_logprobs(prompt, max_tokens)
        return text, mean_lp

    #generate detailed response with token scores
    def generate_with_full_logprobs(
        self,
        prompt: str,
        max_tokens: int = 512,
        return_hidden_states: bool = False,
    ) -> Any:
        torch = self._torch
        import torch.nn.functional as F

        inputs = self._build_input(prompt)
        prompt_len = inputs["input_ids"].shape[1]

        from transformers.generation.utils import GenerateDecoderOnlyOutput
        from transformers.modeling_outputs import BaseModelOutput

        with torch.no_grad():
            raw_out = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                output_scores=True,
                output_hidden_states=return_hidden_states,
                return_dict_in_generate=True,
            )

        del inputs
        torch.cuda.empty_cache()

        out = cast(GenerateDecoderOnlyOutput, raw_out)
        generated_ids = out.sequences[0, prompt_len:]
        scores: tuple = cast(tuple, out.scores or ())

        token_logprobs: List[Tuple[str, float]] = []
        for token_id, score_vec in zip(generated_ids, scores):
            log_probs = F.log_softmax(score_vec[0], dim=-1)
            lp = log_probs[token_id].item()
            tok_str = self._tokenizer.decode(
                [token_id.item()], skip_special_tokens=False
            )
            token_logprobs.append((tok_str, lp))

        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        mean_lp = (
            sum(lp for _, lp in token_logprobs) / len(token_logprobs)
            if token_logprobs else float("nan")
        )

        hidden_states_out: "torch.Tensor | None" = None
        raw_hs = out.hidden_states
        if return_hidden_states and raw_hs is not None:
            hs: tuple = cast(tuple, raw_hs)
            gen_len = len(hs)
            if gen_len > 0:
                num_layers = len(hs[0])
                mid_layer_idx = num_layers // 2
                mid_hidden_states = []
                for step_idx in range(gen_len):
                    step_mid_state = hs[step_idx][mid_layer_idx]
                    token_mid_state = step_mid_state[0, -1, :].detach().cpu().to(torch.float32)
                    mid_hidden_states.append(token_mid_state)
                hidden_states_out = torch.stack(mid_hidden_states)
            else:
                hidden_states_out = torch.empty((0,))

        del out, generated_ids, scores
        torch.cuda.empty_cache()

        if return_hidden_states:
            return text, mean_lp, token_logprobs, hidden_states_out
        return text, mean_lp, token_logprobs

    #generate several randomized answers
    def generate_samples(
        self,
        prompt: str,
        k: int = 5,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 512,
    ) -> List[Tuple[str, float]]:
        torch = self._torch
        import torch.nn.functional as F

        samples: List[Tuple[str, float]] = []

        for i in range(k):
            try:
                inputs = self._build_input(prompt)
                prompt_len = inputs["input_ids"].shape[1]

                with torch.no_grad():
                    raw_out = self._model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )

                from transformers.generation.utils import GenerateDecoderOnlyOutput
                out = cast(GenerateDecoderOnlyOutput, raw_out)
                generated_ids = out.sequences[0, prompt_len:]
                scores: tuple = cast(tuple, out.scores or ())

                lps = []
                for token_id, score_vec in zip(generated_ids, scores):
                    log_probs = F.log_softmax(score_vec[0], dim=-1)
                    lps.append(log_probs[token_id].item())

                text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                mean_lp = sum(lps) / len(lps) if lps else float("nan")
                samples.append((text, mean_lp))

                del inputs, out, generated_ids, scores
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [sample {i+1}/{k} failed: {e}]")
                samples.append(("", float("nan")))

        return samples

#client for qwen instructing models
class LLMClient(_BaseHFClient):
    DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"
    LOG_LABEL = "Qwen2.5"
    GPU_MEMORY_GB = 40

#client for gemma models
class GemmaClient(_BaseHFClient):
    DEFAULT_MODEL = "google/gemma-3-27b-it"
    LOG_LABEL = "Gemma3"
    GPU_MEMORY_GB = 30

#client for qwen text-to-sql smaller model
class TextToSQLClient(_BaseHFClient):
    DEFAULT_MODEL = "Ellbendls/Qwen-2.5-3b-Text_to_SQL"
    LOG_LABEL = "TextToSQL"
    GPU_MEMORY_GB = 10

#client for arctic model
class ArcticText2SQLClient(_BaseHFClient):
    DEFAULT_MODEL = "Snowflake/Arctic-Text2SQL-R1-7B"
    LOG_LABEL = "ArcticSQL"
    GPU_MEMORY_GB = 12