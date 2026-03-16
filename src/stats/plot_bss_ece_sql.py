#done by Sebastian Bastida Marin
#This file calculates and plots calibration metrics like brier skill score and expected calibration error for generated sql results

# Functions:
#compute_ece: calculates the expected calibration error by grouping predictions into bins
#compute_bss: calculates the brier skill score by comparing mean squared error to a base rate
#load_jsonl: reads evaluation data line by line into a list of dictionaries
#plot_grouped_bars: creates a side by side bar chart comparing multiple methods
#main: loads arguments, extracts metrics, computes the scores, prints the output, and builds the chart

import json
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']


#calculates the expected calibration error by grouping predictions into bins
def compute_ece(confidences, outcomes, n_bins=10):
    confidences = np.array(confidences, dtype=float)
    outcomes = np.array(outcomes, dtype=float)
    if len(confidences) == 0:
        return 0.0
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


#calculates the brier skill score by comparing mean squared error to a base rate
def compute_bss(confidences, outcomes):
    confidences = np.array(confidences, dtype=float)
    outcomes = np.array(outcomes, dtype=float)
    if len(confidences) == 0:
        return 0.0
    mse = np.mean(np.square(outcomes - confidences))
    base_rate = np.mean(outcomes)
    brier_ref = base_rate * (1 - base_rate)
    if brier_ref == 0:
        return 0.0
    return 1.0 - (mse / brier_ref)


#reads evaluation data line by line into a list of dictionaries
def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


#creates a side by side bar chart comparing multiple methods
def plot_grouped_bars(ax, method_names, bss_vals, ece_vals, title, show_legend=True):
    x = np.arange(len(method_names))
    bar_width = 0.32

    bss_color = "#6366f1"
    ece_color = "#f43f5e"

    bars1 = ax.bar(x - bar_width / 2, bss_vals, bar_width,
                   label="BSS (higher = better)", color=bss_color,
                   edgecolor="white", linewidth=1.2, zorder=3, alpha=0.92)
    bars2 = ax.bar(x + bar_width / 2, ece_vals, bar_width,
                   label="ECE (lower = better)", color=ece_color,
                   edgecolor="white", linewidth=1.2, zorder=3, alpha=0.92)

    for bar in bars1:
        h = bar.get_height()
        y_pos = h + 0.005 if h >= 0 else h - 0.012
        va = "bottom" if h >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{h:.3f}",
                ha="center", va=va, fontsize=10, fontweight="600", color=bss_color)

    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="600", color=ece_color)

    ax.set_xticks(x)
    ax.set_xticklabels(method_names, fontsize=12, fontweight="500")
    ax.set_ylabel("Score", fontsize=13, fontweight="500")
    ax.set_title(title, fontsize=14, fontweight="700", pad=12)

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


#loads arguments, extracts metrics, computes the scores, prints the output, and builds the chart
def main():
    parser = argparse.ArgumentParser(
        description="Compute BSS & ECE from SQL pipeline JSONL and generate bar chart"
    )
    parser.add_argument("--input", required=True,
                        help="Path to JSONL output from the SQL pipeline")
    parser.add_argument("--output", default=None,
                        help="Output PNG path (default: outputs/bss_ece_sql_chart.png)")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    print(f"Loaded {len(records)} records from {args.input}")

    n_pass = sum(1 for r in records if r["passed"])
    n_fix = sum(1 for r in records if not r["passed"] and r.get("fix_passed"))
    n_fail = len(records) - n_pass - n_fix
    print(f"  Pass: {n_pass}  Fixed: {n_fix}  Fail: {n_fail}  "
          f"EX: {n_pass + n_fix}/{len(records)} ({100*(n_pass+n_fix)/len(records):.1f}%)")

    METHOD_NAMES = ["Tokenized", "Verbalized", "Consistency"]

    #simple helper to build empty data bins
    def empty_bucket():
        return {"confs": [], "outcomes": []}

    line_methods = {m: empty_bucket() for m in METHOD_NAMES}
    token_methods = {m: empty_bucket() for m in METHOD_NAMES}

    #pull probabilities and outcomes from both line level and token level data
    for rec in records:
        for entry in rec.get("line_array", []):
            _, v_conf, t_conf, c_conf, wb_conf, outcome = entry
            line_methods["Verbalized"]["confs"].append(v_conf)
            line_methods["Verbalized"]["outcomes"].append(outcome)
            line_methods["Tokenized"]["confs"].append(t_conf)
            line_methods["Tokenized"]["outcomes"].append(outcome)
            line_methods["Consistency"]["confs"].append(c_conf)
            line_methods["Consistency"]["outcomes"].append(outcome)

        for entry in rec.get("token_array", []):
            _, v_conf, t_conf, c_conf, wb_conf, outcome = entry
            token_methods["Verbalized"]["confs"].append(v_conf)
            token_methods["Verbalized"]["outcomes"].append(outcome)
            token_methods["Tokenized"]["confs"].append(t_conf)
            token_methods["Tokenized"]["outcomes"].append(outcome)
            token_methods["Consistency"]["confs"].append(c_conf)
            token_methods["Consistency"]["outcomes"].append(outcome)

    n_line_pts = len(line_methods["Tokenized"]["confs"])
    n_tok_pts = len(token_methods["Tokenized"]["confs"])

    print(f"BSS & ECE Results  ({len(records)} BIRD SQL examples)")
    print(f"Line-level data points:  {n_line_pts}")
    print(f"Token-level data points: {n_tok_pts}")

    if n_line_pts > 0:
        print(f"Line-level base rate (% correct): "
              f"{np.mean(line_methods['Tokenized']['outcomes']):.3f}")
    if n_tok_pts > 0:
        print(f"Token-level base rate (% correct): "
              f"{np.mean(token_methods['Tokenized']['outcomes']):.3f}")

    line_bss, line_ece = [], []
    token_bss, token_ece = [], []

    print(f"\n  LINE-LEVEL:")
    print(f"{'Method':<15} {'BSS':>8} {'ECE':>8} {'N':>8}")
    for m in METHOD_NAMES:
        bss = compute_bss(line_methods[m]["confs"], line_methods[m]["outcomes"])
        ece = compute_ece(line_methods[m]["confs"], line_methods[m]["outcomes"])
        n = len(line_methods[m]["confs"])
        line_bss.append(bss)
        line_ece.append(ece)
        print(f"  {m:<15} {bss:>+8.4f} {ece:>8.4f} {n:>8}")

    print(f"\n  TOKEN-LEVEL:")
    print(f"{'Method':<15} {'BSS':>8} {'ECE':>8} {'N':>8}")
    for m in METHOD_NAMES:
        bss = compute_bss(token_methods[m]["confs"], token_methods[m]["outcomes"])
        ece = compute_ece(token_methods[m]["confs"], token_methods[m]["outcomes"])
        n = len(token_methods[m]["confs"])
        token_bss.append(bss)
        token_ece.append(ece)
        print(f"  {m:<15} {bss:>+8.4f} {ece:>8.4f} {n:>8}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("#ffffff")

    plot_grouped_bars(ax1, METHOD_NAMES, line_bss, line_ece,
                      f"Line-Level  ({n_line_pts} lines)", show_legend=True)
    plot_grouped_bars(ax2, METHOD_NAMES, token_bss, token_ece,
                      f"Token-Level  ({n_tok_pts} tokens)", show_legend=False)

    fig.suptitle(
        f"BSS & ECE — BIRD SQL ({len(records)} examples, "
        f"EX {100*(n_pass+n_fix)/len(records):.1f}%)",
        fontsize=16, fontweight="800", y=1.02, color="#1f2937"
    )

    plt.tight_layout()

    out_path = args.output or "outputs/bss_ece_sql_chart.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#ffffff")
    print(f"\nChart saved -> {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
