#done by Rei Shindo + Logan Mifflin
#This file makes bar charts showing bss and ece scores for datasets

# Functions:
#_sigmoid maps numbers to probabilities between zero and one
#apply_platt_scaling scales confidence scores to be more accurate
#compute_ece calculates expected calibration error
#compute_bss calculates the brier skill score
#compute_bss_kfold calculates the brier skill score using cross validation
#load_jsonl reads json files line by line
#infer_dataset_label figures out the dataset name from the file
#infer_run_label figures out the run name from the file
#collect_method_data grabs confidence scores and outcomes from records
#collect_whitebox_method_data grabs whitebox specific data and normal data
#collect_whitebox_only grabs only the whitebox specific data
#compute_metrics calculates bss and ece for all given methods
#plot_grouped_bars draws the actual bar chart for a dataset
#plot_grouped_bars_two_runs draws a bar chart comparing two different runs
#plot_whitebox_only_two_runs draws a chart just for two whitebox runs
#main handles command line arguments and runs the entire process

import json
import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from scipy.optimize import minimize
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

#this maps numbers to probabilities between zero and one
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

#this scales confidence scores to be more accurate
def apply_platt_scaling(confidences, outcomes, n_splits=5, method="platt"):
    confidences = np.array(confidences, dtype=float)
    outcomes    = np.array(outcomes,    dtype=float)

    if len(confidences) == 0:
        return confidences

    X      = confidences.reshape(-1, 1)
    scaled = np.zeros_like(confidences)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X):
        c_train, y_train = confidences[train_idx], outcomes[train_idx]
        c_test           = confidences[test_idx]

        if len(np.unique(y_train)) < 2:
            scaled[test_idx] = c_test
            continue

        if method == "platt":
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(c_train.reshape(-1, 1), y_train)
            scaled[test_idx] = lr.predict_proba(c_test.reshape(-1, 1))[:, 1]

        elif method == "platt-balanced":
            lr = LogisticRegression(class_weight="balanced",
                                    max_iter=1000, random_state=42)
            lr.fit(c_train.reshape(-1, 1), y_train)
            scaled[test_idx] = lr.predict_proba(c_test.reshape(-1, 1))[:, 1]

        elif method == "isotonic":
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(c_train, y_train)
            scaled[test_idx] = ir.predict(c_test)

        elif method == "mse":
            def brier(params):
                a, b = params
                pred = _sigmoid(a * c_train + b)
                return np.mean((y_train - pred) ** 2)

            res = minimize(brier, x0=[1.0, 0.0], method="L-BFGS-B")
            a, b = res.x
            scaled[test_idx] = _sigmoid(a * c_test + b)

        else:
            raise ValueError(f"Unknown scale method: {method!r}")

    return scaled


#this calculates expected calibration error
def compute_ece(confidences, outcomes, n_bins=10):
    confidences = np.array(confidences, dtype=float)
    outcomes = np.array(outcomes, dtype=float)
    if len(confidences) == 0: return 0.0
    
    buckets = np.clip((confidences * n_bins).astype(int), 0, n_bins - 1)
    
    ece = 0.0
    for i in range(n_bins):
        mask = (buckets == i)
        if np.any(mask):
            bucket_conf = confidences[mask].mean()
            bucket_acc = outcomes[mask].mean()
            weight = mask.sum() / len(confidences)
            ece += weight * abs(bucket_acc - bucket_conf)
    return ece


#this calculates the brier skill score
def compute_bss(confidences, outcomes):
    confidences = np.array(confidences, dtype=float)
    outcomes = np.array(outcomes, dtype=float)
    size = len(confidences)
    if size == 0: return 0.0

    mse = np.mean(np.square(outcomes - confidences))
    base = np.mean(outcomes)
    brier_ref = base * (1 - base)
    if brier_ref == 0:
        return 0.0
    return 1.0 - (mse / brier_ref)


#this calculates the brier skill score using cross validation
def compute_bss_kfold(raw_confs, outcomes, n_splits=5, method="platt"):
    raw_confs = np.array(raw_confs, dtype=float)
    outcomes  = np.array(outcomes,  dtype=float)
    if len(raw_confs) == 0:
        return 0.0

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_bss = []

    for train_idx, test_idx in kf.split(raw_confs):
        c_train, y_train = raw_confs[train_idx], outcomes[train_idx]
        c_test,  y_test  = raw_confs[test_idx],  outcomes[test_idx]

        if len(np.unique(y_train)) < 2:
            #this skips scaling if only one class is present
            scaled_test = c_test
        else:
            if method == "platt":
                lr = LogisticRegression(max_iter=1000, random_state=42)
                lr.fit(c_train.reshape(-1, 1), y_train)
                scaled_test = lr.predict_proba(c_test.reshape(-1, 1))[:, 1]
            elif method == "platt-balanced":
                lr = LogisticRegression(class_weight="balanced",
                                        max_iter=1000, random_state=42)
                lr.fit(c_train.reshape(-1, 1), y_train)
                scaled_test = lr.predict_proba(c_test.reshape(-1, 1))[:, 1]
            elif method == "isotonic":
                ir = IsotonicRegression(out_of_bounds="clip")
                ir.fit(c_train, y_train)
                scaled_test = ir.predict(c_test)
            elif method == "mse":
                def brier_loss(params):
                    a, b = params
                    return np.mean((y_train - _sigmoid(a * c_train + b)) ** 2)
                res = minimize(brier_loss, x0=[1.0, 0.0], method="L-BFGS-B")
                scaled_test = _sigmoid(res.x[0] * c_test + res.x[1])
            else:
                scaled_test = c_test

        fold_bss.append(compute_bss(scaled_test, y_test))

    return float(np.mean(fold_bss))


#this reads json files line by line
def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '{':
            try:
                data = json.load(f)
                if isinstance(data, dict):
                    for key in data:
                        if isinstance(data[key], list):
                            records.extend(data[key])
                elif isinstance(data, list):
                    records = data
            except json.JSONDecodeError:
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        elif first_char == '[':
            records = json.load(f)
        else:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


#this figures out the dataset name from the file
def infer_dataset_label(records, input_path):
    if records:
        ds = records[0].get("dataset")
        if isinstance(ds, str) and ds.strip():
            return ds.upper()

    name = os.path.basename(input_path).lower()
    if "humaneval" in name:
        return "HUMANEVAL"
    if "mbpp" in name:
        return "MBPP"
    return os.path.splitext(os.path.basename(input_path))[0]


#this figures out the run name from the file
def infer_run_label(input_path, idx):
    name = os.path.basename(input_path).lower()
    if "first" in name or "run1" in name or "run_1" in name:
        return "First probe"
    if "second" in name or "run2" in name or "run_2" in name:
        return "Second probe"
    return f"Run {idx + 1}"


#this grabs confidence scores and outcomes from records
def collect_method_data(records):
    line_methods = {
        "Verbalized": {"confs": [], "outcomes": []},
        "Tokenized": {"confs": [], "outcomes": []},
        "Consistency": {"confs": [], "outcomes": []},
    }
    token_methods = {
        "Verbalized": {"confs": [], "outcomes": []},
        "Tokenized": {"confs": [], "outcomes": []},
        "Consistency": {"confs": [], "outcomes": []},
    }

    def _extract_entry(entry):
        if not isinstance(entry, (list, tuple)):
            return None

        if len(entry) >= 6:
            _, v_conf, t_conf, c_conf, _wb_conf, in_final = entry[:6]
        elif len(entry) >= 5:
            _, v_conf, t_conf, c_conf, in_final = entry[:5]
        else:
            return None

        return v_conf, t_conf, c_conf, in_final

    for rec in records:
        for entry in rec.get("line_array", []):
            parsed = _extract_entry(entry)
            if parsed is None:
                continue
            v_conf, t_conf, c_conf, in_final = parsed
            line_methods["Verbalized"]["confs"].append(v_conf)
            line_methods["Verbalized"]["outcomes"].append(in_final)
            line_methods["Tokenized"]["confs"].append(t_conf)
            line_methods["Tokenized"]["outcomes"].append(in_final)
            line_methods["Consistency"]["confs"].append(c_conf)
            line_methods["Consistency"]["outcomes"].append(in_final)

        for entry in rec.get("token_array", []):
            parsed = _extract_entry(entry)
            if parsed is None:
                continue
            v_conf, t_conf, c_conf, in_final = parsed
            token_methods["Verbalized"]["confs"].append(v_conf)
            token_methods["Verbalized"]["outcomes"].append(in_final)
            token_methods["Tokenized"]["confs"].append(t_conf)
            token_methods["Tokenized"]["outcomes"].append(in_final)
            token_methods["Consistency"]["confs"].append(c_conf)
            token_methods["Consistency"]["outcomes"].append(in_final)

    return line_methods, token_methods


#this grabs whitebox specific data and normal data
def collect_whitebox_method_data(records):
    methods = ["Tokenized", "Verbalized", "Consistency", "Whitebox"]
    line_methods = {m: {"confs": [], "outcomes": []} for m in methods}
    token_methods = {m: {"confs": [], "outcomes": []} for m in methods}

    def _extract_entry(entry):
        """Return (v_conf, t_conf, c_conf, wb_conf, outcome) or None."""
        if not isinstance(entry, (list, tuple)):
            return None

        if len(entry) >= 6:
            _, v_conf, t_conf, c_conf, wb_conf, outcome = entry[:6]
        elif len(entry) >= 5:
            _, v_conf, t_conf, c_conf, outcome = entry[:5]
            wb_conf = None
        else:
            return None

        return v_conf, t_conf, c_conf, wb_conf, outcome

    def _append_methods(target, parsed):
        v_conf, t_conf, c_conf, wb_conf, outcome = parsed
        target["Verbalized"]["confs"].append(v_conf)
        target["Verbalized"]["outcomes"].append(outcome)
        target["Tokenized"]["confs"].append(t_conf)
        target["Tokenized"]["outcomes"].append(outcome)
        target["Consistency"]["confs"].append(c_conf)
        target["Consistency"]["outcomes"].append(outcome)
        if wb_conf is not None:
            target["Whitebox"]["confs"].append(wb_conf)
            target["Whitebox"]["outcomes"].append(outcome)

    for rec in records:
        for entry in rec.get("line_array", []):
            parsed = _extract_entry(entry)
            if parsed is not None:
                _append_methods(line_methods, parsed)

        for entry in rec.get("token_array", []):
            parsed = _extract_entry(entry)
            if parsed is not None:
                _append_methods(token_methods, parsed)

    return line_methods, token_methods


#this grabs only the whitebox specific data
def collect_whitebox_only(records):
    line_methods, token_methods = collect_whitebox_method_data(records)
    return line_methods["Whitebox"], token_methods["Whitebox"]


#this calculates bss and ece for all given methods
def compute_metrics(methods, method_names, scale_method=None):
    bss_vals, ece_vals = [], []

    if scale_method:
        for m in method_names:
            methods[m]["raw_confs"] = list(methods[m]["confs"])
            methods[m]["confs"] = apply_platt_scaling(
                methods[m]["confs"], methods[m]["outcomes"], method=scale_method
            ).tolist()

    for m in method_names:
        if scale_method:
            bss = compute_bss_kfold(
                methods[m]["raw_confs"], methods[m]["outcomes"], method=scale_method
            )
        else:
            bss = compute_bss(methods[m]["confs"], methods[m]["outcomes"])
        ece = compute_ece(methods[m]["confs"], methods[m]["outcomes"])
        bss_vals.append(bss)
        ece_vals.append(ece)

    return bss_vals, ece_vals


#this draws the actual bar chart for a dataset
def plot_grouped_bars(ax, method_names, bss_vals, ece_vals, title, show_legend=True):
    x = np.arange(len(method_names))
    bar_width = 0.32

    bss_color = "#6366f1"
    ece_color = "#f43f5e"

    bars1 = ax.bar(x - bar_width/2, bss_vals, bar_width,
                   label="BSS (higher = better)", color=bss_color,
                   edgecolor="white", linewidth=1.2, zorder=3, alpha=0.92)
    bars2 = ax.bar(x + bar_width/2, ece_vals, bar_width,
                   label="ECE (lower = better)", color=ece_color,
                   edgecolor="white", linewidth=1.2, zorder=3, alpha=0.92)

    for bar in bars1:
        h = bar.get_height()
        y_pos = h + 0.005 if h >= 0 else h - 0.012
        va = "bottom" if h >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f"{h:.3f}",
                ha="center", va=va, fontsize=11, fontweight="600", color=bss_color)

    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="600", color=ece_color)

    ax.set_xticks(x)
    ax.set_xticklabels(method_names, fontsize=13, fontweight="500")
    ax.set_ylabel("Score", fontsize=13, fontweight="500")
    ax.set_title(title, fontsize=15, fontweight="700", pad=12)

    if show_legend:
        ax.legend(fontsize=11, loc="best", framealpha=0.9,
                  edgecolor="#e5e7eb", fancybox=True)

    ax.axhline(y=0, color="#9ca3af", linewidth=0.8, linestyle="--", zorder=1)

    all_vals = list(bss_vals) + list(ece_vals)
    y_min = min(min(all_vals), 0) - 0.06
    y_max = max(all_vals) + 0.06
    ax.set_ylim(y_min, y_max)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d1d5db")
    ax.spines["bottom"].set_color("#d1d5db")
    ax.tick_params(colors="#6b7280")
    ax.yaxis.label.set_color("#374151")
    ax.set_facecolor("#fafafa")
    ax.grid(axis="y", alpha=0.3, color="#d1d5db", zorder=0)


#this draws a bar chart comparing two different runs
def plot_grouped_bars_two_runs(
    ax,
    method_names,
    run1_bss,
    run1_ece,
    run2_bss,
    run2_ece,
    run1_label,
    run2_label,
    title,
    show_legend=True,
):
    x = np.arange(len(method_names))
    bar_width = 0.18

    bss_color_1 = "#4f46e5"
    bss_color_2 = "#818cf8"
    ece_color_1 = "#e11d48"
    ece_color_2 = "#fb7185"

    bars_r1_bss = ax.bar(
        x - 1.5 * bar_width,
        run1_bss,
        bar_width,
        label=f"{run1_label} - BSS",
        color=bss_color_1,
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
        alpha=0.94,
    )
    bars_r2_bss = ax.bar(
        x - 0.5 * bar_width,
        run2_bss,
        bar_width,
        label=f"{run2_label} - BSS",
        color=bss_color_2,
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
        alpha=0.94,
    )
    bars_r1_ece = ax.bar(
        x + 0.5 * bar_width,
        run1_ece,
        bar_width,
        label=f"{run1_label} - ECE",
        color=ece_color_1,
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
        alpha=0.94,
    )
    bars_r2_ece = ax.bar(
        x + 1.5 * bar_width,
        run2_ece,
        bar_width,
        label=f"{run2_label} - ECE",
        color=ece_color_2,
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
        alpha=0.94,
    )

    for bars, color in ((bars_r1_bss, bss_color_1), (bars_r2_bss, bss_color_2),
                        (bars_r1_ece, ece_color_1), (bars_r2_ece, ece_color_2)):
        for bar in bars:
            h = bar.get_height()
            y_pos = h + 0.005 if h >= 0 else h - 0.012
            va = "bottom" if h >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                f"{h:.3f}",
                ha="center",
                va=va,
                fontsize=9,
                fontweight="600",
                color=color,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(method_names, fontsize=13, fontweight="500")
    ax.set_ylabel("Score", fontsize=13, fontweight="500")
    ax.set_title(title, fontsize=15, fontweight="700", pad=12)

    if show_legend:
        ax.legend(fontsize=10, loc="best", framealpha=0.9,
                  edgecolor="#e5e7eb", fancybox=True)

    ax.axhline(y=0, color="#9ca3af", linewidth=0.8, linestyle="--", zorder=1)

    all_vals = list(run1_bss) + list(run1_ece) + list(run2_bss) + list(run2_ece)
    y_min = min(min(all_vals), 0) - 0.06
    y_max = max(all_vals) + 0.06
    ax.set_ylim(y_min, y_max)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d1d5db")
    ax.spines["bottom"].set_color("#d1d5db")
    ax.tick_params(colors="#6b7280")
    ax.yaxis.label.set_color("#374151")
    ax.set_facecolor("#fafafa")
    ax.grid(axis="y", alpha=0.3, color="#d1d5db", zorder=0)


#this draws a chart just for two whitebox runs
def plot_whitebox_only_two_runs(
    ax,
    run1_bss,
    run1_ece,
    run2_bss,
    run2_ece,
    title,
    show_legend=True,
):
    labels = ["First Whitebox", "Second Whitebox"]
    x = np.arange(len(labels))
    bar_width = 0.34

    bss_vals = [run1_bss, run2_bss]
    ece_vals = [run1_ece, run2_ece]

    bss_color = "#4f46e5"
    ece_color = "#e11d48"

    bars_bss = ax.bar(
        x - bar_width / 2,
        bss_vals,
        bar_width,
        label="BSS (higher = better)",
        color=bss_color,
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
        alpha=0.94,
    )
    bars_ece = ax.bar(
        x + bar_width / 2,
        ece_vals,
        bar_width,
        label="ECE (lower = better)",
        color=ece_color,
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
        alpha=0.94,
    )

    for bars, color in ((bars_bss, bss_color), (bars_ece, ece_color)):
        for bar in bars:
            h = bar.get_height()
            y_pos = h + 0.005 if h >= 0 else h - 0.012
            va = "bottom" if h >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                f"{h:.3f}",
                ha="center",
                va=va,
                fontsize=10,
                fontweight="600",
                color=color,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13, fontweight="500")
    ax.set_ylabel("Score", fontsize=13, fontweight="500")
    ax.set_title(title, fontsize=15, fontweight="700", pad=12)

    if show_legend:
        ax.legend(fontsize=10, loc="best", framealpha=0.9,
                  edgecolor="#e5e7eb", fancybox=True)

    ax.axhline(y=0, color="#9ca3af", linewidth=0.8, linestyle="--", zorder=1)

    all_vals = bss_vals + ece_vals
    y_min = min(min(all_vals), 0) - 0.06
    y_max = max(all_vals) + 0.06
    ax.set_ylim(y_min, y_max)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d1d5db")
    ax.spines["bottom"].set_color("#d1d5db")
    ax.tick_params(colors="#6b7280")
    ax.yaxis.label.set_color("#374151")
    ax.set_facecolor("#fafafa")
    ax.grid(axis="y", alpha=0.3, color="#d1d5db", zorder=0)


#this handles command line arguments and runs the entire process
def main():
    parser = argparse.ArgumentParser(
        description="Compute BSS & ECE from pipeline JSONL and generate bar chart"
    )
    parser.add_argument("--input", nargs="+", required=True,
                        help="One or more JSONL/grouped JSON files from the pipeline")
    parser.add_argument("--output", default=None,
                        help="Output PNG path (default: auto-named in outputs/)")
    parser.add_argument("--scale-method",
                        choices=["platt", "platt-balanced", "isotonic", "mse"],
                        default=None,
                        help=("Apply confidence rescaling before metrics. "
                              "Choices: platt | platt-balanced | isotonic | mse"))
    parser.add_argument("--compare-runs", action="store_true",
                        help=("Compare exactly two inputs as first/second probe in a single "
                              "image with side-by-side bars."))
    parser.add_argument("--run-labels", nargs=2, metavar=("RUN1", "RUN2"),
                        default=None,
                        help=("Labels for --compare-runs (default: inferred or "
                              "'First probe'/'Second probe')."))
    parser.add_argument("--whitebox-input", default=None,
                        help=("Path to a whitebox run file. When provided (without "
                              "--compare-runs), Whitebox from this file is added to "
                              "the chart alongside Tokenized/Verbalized/Consistency "
                              "from --input."))
    args = parser.parse_args()

    if args.compare_runs and len(args.input) != 2:
        raise ValueError("--compare-runs requires exactly two --input files")
    if args.whitebox_input and args.compare_runs:
        raise ValueError("--whitebox-input cannot be used with --compare-runs")
    if args.whitebox_input and len(args.input) != 1:
        raise ValueError("--whitebox-input currently supports exactly one --input file")

    method_names = ["Tokenized", "Verbalized", "Consistency"]
    wb_line, wb_token = None, None
    if args.whitebox_input:
        wb_records = load_jsonl(args.whitebox_input)
        wb_line, wb_token = collect_whitebox_only(wb_records)
        method_names = ["Tokenized", "Verbalized", "Consistency", "Whitebox"]
        print(f"Loaded {len(wb_records)} records from {args.whitebox_input}")
    scale_label = f"Scaled: {args.scale_method}" if args.scale_method else "Unscaled"
    scale_suffix = f" [{scale_label}]" if args.scale_method else ""

    dataset_results = []

    for input_path in args.input:
        records = load_jsonl(input_path)
        dataset_label = infer_dataset_label(records, input_path)
        print(f"Loaded {len(records)} records from {input_path}")

        n_pass = sum(1 for r in records if r.get("passed"))
        n_fix = sum(1 for r in records if not r.get("passed") and r.get("fix_passed"))
        n_fail = len(records) - n_pass - n_fix
        print(f"  Dataset: {dataset_label}")
        print(f"  Pass: {n_pass}  Fixed: {n_fix}  Fail: {n_fail}")

        if args.compare_runs:
            method_names = ["Tokenized", "Verbalized", "Consistency", "Whitebox"]
            line_methods, token_methods = collect_whitebox_method_data(records)
        else:
            line_methods, token_methods = collect_method_data(records)
            if args.whitebox_input and wb_line is not None and wb_token is not None:
                line_methods["Whitebox"] = {
                    "confs": list(wb_line["confs"]),
                    "outcomes": list(wb_line["outcomes"]),
                }
                token_methods["Whitebox"] = {
                    "confs": list(wb_token["confs"]),
                    "outcomes": list(wb_token["outcomes"]),
                }

        n_line_pts = len(line_methods["Tokenized"]["confs"])
        n_tok_pts = len(token_methods["Tokenized"]["confs"])

        print(f"  BSS & ECE Results  ({dataset_label}, {len(records)} problems)")
        print(f"  Line-level data points: {n_line_pts}")
        print(f"  Token-level data points: {n_tok_pts}")

        if n_line_pts > 0:
            line_base = np.mean(line_methods["Tokenized"]["outcomes"])
            print(f"  Line-level base rate (% kept): {line_base:.3f}")
        if n_tok_pts > 0:
            tok_base = np.mean(token_methods["Tokenized"]["outcomes"])
            print(f"  Token-level base rate (% kept): {tok_base:.3f}")

        if args.scale_method:
            print(f"\n  Applying rescaling: {args.scale_method} (5-fold CV)…")

        line_bss, line_ece = compute_metrics(
            line_methods, method_names, scale_method=args.scale_method
        )
        token_bss, token_ece = compute_metrics(
            token_methods, method_names, scale_method=args.scale_method
        )

        print(f"\n  LINE-LEVEL ({scale_label}):")
        print(f"  {'Method':<15} {'BSS':>8} {'ECE':>8} {'N':>6}")
        print(f"  {'-'*40}")
        for i, m in enumerate(method_names):
            n = len(line_methods[m]["confs"])
            print(f"  {m:<15} {line_bss[i]:>+8.4f} {line_ece[i]:>8.4f} {n:>6}")

        print(f"\n  TOKEN-LEVEL ({scale_label}):")
        print(f"  {'Method':<15} {'BSS':>8} {'ECE':>8} {'N':>6}")
        print(f"  {'-'*40}")
        for i, m in enumerate(method_names):
            n = len(token_methods[m]["confs"])
            print(f"  {m:<15} {token_bss[i]:>+8.4f} {token_ece[i]:>8.4f} {n:>6}")

        dataset_results.append({
            "label": dataset_label,
            "n_problems": len(records),
            "n_line_pts": n_line_pts,
            "n_tok_pts": n_tok_pts,
            "line_bss": line_bss,
            "line_ece": line_ece,
            "token_bss": token_bss,
            "token_ece": token_ece,
        })

    if args.compare_runs:
        #this extracts data from the two compared runs
        run1 = dataset_results[0]
        run2 = dataset_results[1]

        if args.run_labels:
            run1_label, run2_label = args.run_labels
        else:
            run1_label = infer_run_label(args.input[0], 0)
            run2_label = infer_run_label(args.input[1], 1)

        fig, (ax_line, ax_token) = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor("#ffffff")

        wb_idx = method_names.index("Whitebox")
        plot_whitebox_only_two_runs(
            ax_line,
            run1["line_bss"][wb_idx],
            run1["line_ece"][wb_idx],
            run2["line_bss"][wb_idx],
            run2["line_ece"][wb_idx],
            f"Line-Level ({run1['n_line_pts']} lines){scale_suffix}",
            show_legend=True,
        )
        plot_whitebox_only_two_runs(
            ax_token,
            run1["token_bss"][wb_idx],
            run1["token_ece"][wb_idx],
            run2["token_bss"][wb_idx],
            run2["token_ece"][wb_idx],
            f"Token-Level ({run1['n_tok_pts']} tokens){scale_suffix}",
            show_legend=False,
        )

        fig.suptitle(
            f"BSS & ECE - BIRD ({run1_label} vs {run2_label}){scale_suffix}",
            fontsize=17,
            fontweight="800",
            y=1.02,
            color="#1f2937",
        )

        plt.tight_layout()

        if args.output:
            out_path = args.output
        else:
            suffix = f"_{args.scale_method.replace('-', '_')}" if args.scale_method else ""
            out_path = f"outputs/bss_ece_whitebox_compare{suffix}.png"

        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#ffffff")
        print(f"\nChart saved -> {out_path}")
        plt.show()
        return

    n_datasets = len(dataset_results)
    #this sets up the matplotlib figure and subplots
    fig, axes = plt.subplots(n_datasets, 2, figsize=(14, 6 * n_datasets))
    fig.patch.set_facecolor("#ffffff")

    if n_datasets == 1:
        axes = np.array([axes])

    for i, ds in enumerate(dataset_results):
        ax_line, ax_token = axes[i]
        show_legend = (i == 0)
        is_bird = ("bird" in ds["label"].lower())
        line_title = (
            f"BIRD - Line-Level ({ds['n_line_pts']} lines)"
            if is_bird else
            f"{ds['label']} - Line-Level ({ds['n_line_pts']} lines){scale_suffix}"
        )
        token_title = (
            f"BIRD - Token-Level ({ds['n_tok_pts']} tokens)"
            if is_bird else
            f"{ds['label']} - Token-Level ({ds['n_tok_pts']} tokens){scale_suffix}"
        )
        plot_grouped_bars(
            ax_line, method_names, ds["line_bss"], ds["line_ece"],
            line_title,
            show_legend=show_legend,
        )
        plot_grouped_bars(
            ax_token, method_names, ds["token_bss"], ds["token_ece"],
            token_title,
            show_legend=False,
        )

    dataset_title = " + ".join(ds["label"] for ds in dataset_results)
    if len(dataset_results) == 1 and ("bird" in dataset_results[0]["label"].lower()):
        super_title = "BSS & ECE - BIRD"
    else:
        super_title = f"BSS & ECE - {dataset_title}{scale_suffix}"
    fig.suptitle(
        super_title,
        fontsize=17,
        fontweight="800",
        y=1.01,
        color="#1f2937",
    )

    plt.tight_layout()

    if args.output:
        out_path = args.output
    else:
        suffix = f"_{args.scale_method.replace('-', '_')}" if args.scale_method else ""
        if len(dataset_results) == 1:
            dataset_slug = dataset_results[0]["label"].lower()
            wb_suffix = "_with_whitebox" if args.whitebox_input else ""
            out_path = f"outputs/bss_ece_chart_{dataset_slug}{wb_suffix}{suffix}.png"
        else:
            out_path = f"outputs/bss_ece_chart_multi{suffix}.png"

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#ffffff")
    print(f"\nChart saved -> {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
