#done by Logan Mifflin
#fits and saves a whitebox probe from collected data
#loads data, trains, calibrates, and writes a probe file

import argparse, sys, time, random
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.whitebox_probing import WhiteBoxProbe, train_probe_model, calibrate_probe


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(
        description="save a WhiteBoxProbe from collected training data"
    )
    p.add_argument("--training_data", required=True,
                   help="Path to .pt file produced by train_probe.py")
    p.add_argument("--output", default=None,
                   help="Output probe .pt path (default: outputs/probe_trained_<ts>.pt)")
    p.add_argument("--epochs", type=int, default=10,
                   help="Training epochs (default: 10)")
    p.add_argument("--lr", type=float, default=0.001,
                   help="Learning rate (default: 0.001)")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Gradient accumulation batch size (default: 32)")
    p.add_argument("--val_split", type=float, default=0.2,
                   help="Fraction of data held out for Platt calibration (default: 0.2)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    #set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_path = Path(args.training_data)
    if not train_path.exists():
        print(f"ERROR: training data file not found: {train_path}"); sys.exit(1)

    _log(f"Loading training data from {train_path}…")
    checkpoint = torch.load(train_path, map_location="cpu", weights_only=True)

    hidden_states: torch.Tensor = checkpoint["hidden_states"]
    labels: torch.Tensor = checkpoint["labels"]
    pca_mean: torch.Tensor = checkpoint["pca_mean"]
    pca_components: torch.Tensor = checkpoint["pca_components"]
    input_dim: int = int(checkpoint["input_dim"])
    proj_dim: int = int(checkpoint["proj_dim"])
    base_model = checkpoint.get("base_model", "Unknown")

    size_hs = hidden_states.size(0)
    size_lb = labels.size(0)
    if size_hs != size_lb:
        #trim if sizes do not match
        msg = f"WARNING: hidden_states ({size_hs}) and labels ({size_lb}) sizes do NOT match!"
        min_size = min(size_hs, size_lb)
        print(f"  [{time.strftime('%H:%M:%S')}] {msg} Truncating to {min_size}.")
        hidden_states = hidden_states[:min_size]
        labels = labels[:min_size]

    n_pos = int(labels.sum().item())
    n_neg = len(labels) - n_pos

    print(f"  [{time.strftime('%H:%M:%S')}] Loaded {len(labels)} tokens  ({n_pos} correct / {n_neg} incorrect)")
    print(f"  [{time.strftime('%H:%M:%S')}] Base Model: {base_model}")
    _log(f"input_dim={input_dim}, proj_dim={proj_dim}")

    if len(labels) == 0:
        print("ERROR: training data is empty."); sys.exit(1)

    pos_idx = (labels == 1).nonzero(as_tuple=True)[0].tolist()
    neg_idx = (labels == 0).nonzero(as_tuple=True)[0].tolist()
    
    #balance both classes
    min_count = min(len(pos_idx), len(neg_idx))
    
    if min_count == 0:
        print("ERROR: Cannot balance data. One of the classes (0 or 1) has zero samples."); sys.exit(1)
        
    random.shuffle(pos_idx)
    random.shuffle(neg_idx)
    
    balanced_pos_idx = pos_idx[:min_count]
    balanced_neg_idx = neg_idx[:min_count]
    
    balanced_indices = balanced_pos_idx + balanced_neg_idx
    random.shuffle(balanced_indices)
    
    N_balanced = len(balanced_indices)
    print(f"  [{time.strftime('%H:%M:%S')}] Balanced data to {N_balanced} tokens ({min_count} correct / {min_count} incorrect)")
    
    #split train and val
    split = max(1, int(N_balanced * (1 - args.val_split)))
    train_idx = balanced_indices[:split]
    val_idx   = balanced_indices[split:]

    train_data = [(hidden_states[i].unsqueeze(0), int(labels[i].item()))
                  for i in train_idx]
    val_data   = [(hidden_states[i].unsqueeze(0), int(labels[i].item()))
                  for i in val_idx]

    _log(f"Train: {len(train_data)} tokens | Val: {len(val_data)} tokens")

    _log(f"Training probe (epochs={args.epochs}, lr={args.lr})…")
    t0 = time.time()
    probe = train_probe_model(
        train_data=train_data,
        input_dim=proj_dim,
        proj_dim=proj_dim,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )
    _log(f"Training done in {time.time() - t0:.1f}s")

    if val_data:
        _log(f"Calibrating with Platt scaling on {len(val_data)} val tokens…")
        calibrate_probe(probe, val_data)
        _log("Calibration done.")
    else:
        _log("No validation data — skipping Platt calibration.")

    #pick output path
    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).parent.parent / "outputs"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"probe_trained_{ts}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    probe.save_with_pca(str(out_path), pca_mean=pca_mean, pca_components=pca_components,
                        raw_input_dim=input_dim)

    _log(f"Probe saved: {out_path}")

    #quick reload check
    _log("Sanity check: loading probe back…")
    loaded = WhiteBoxProbe.load(str(out_path))
    assert loaded.input_dim == proj_dim, "input_dim mismatch after reload"
    assert hasattr(loaded, "pca_mean") and loaded.pca_mean is not None, "PCA mean missing"
    _log(f"Probe loaded OK. pca_mean shape={loaded.pca_mean.shape}, ")

    print(f"\nDone! Trained probe saved to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
