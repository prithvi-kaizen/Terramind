"""Command-line adapter; compatibility imports for existing callers."""
import argparse
from methanepulse.runtime import run_inference, get_device
from methanepulse.model import TerraMindMethanePulse
from methanepulse.preprocessing import load_and_preprocess
from methanepulse.artifacts import plume_centroids

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MethanePulse inference — Sentinel-2 methane plume detection"
    )
    parser.add_argument("--input",      required=True)
    parser.add_argument("--checkpoint", default="terramind_methane_best.pt")
    parser.add_argument("--output",     default="outputs/inference/")
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    args = parser.parse_args()
    run_inference(args.input, args.checkpoint, args.output, args.threshold, device=args.device)
