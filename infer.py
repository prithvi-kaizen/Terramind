"""Command-line adapter; compatibility imports for existing callers."""
import argparse
from methanepulse.runtime import run_inference, get_device
from methanepulse.candidates import CandidateConfig, FILTER_FLAGS
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
    parser.add_argument("--min-area-m2", type=float, default=1600)
    parser.add_argument("--filter-flags", nargs="*", choices=FILTER_FLAGS, default=[])
    args = parser.parse_args()
    run_inference(args.input, args.checkpoint, args.output, args.threshold, device=args.device, candidate_config=CandidateConfig(min_area_m2=args.min_area_m2, filter_flags=tuple(args.filter_flags)))
