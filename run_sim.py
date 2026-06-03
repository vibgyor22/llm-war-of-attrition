"""Quick smoke run: 2 sims per condition across all episodes."""
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
sys.path.insert(0, str(Path(__file__).parent))

from src.simulation.batch_runner import run_all, compile_results

print("Starting batch run: 2 sims/condition ...", flush=True)
try:
    df = run_all(n_sims_per_condition=2, output_dir=Path("outputs"))
    print(f"Batch complete: {len(df)} simulations", flush=True)
    print(df[["episode_id", "condition_id", "winner", "concession_period"]].to_string(), flush=True)
except Exception as e:
    print(f"Batch error: {e}", flush=True)
    print("Compiling available transcripts...", flush=True)
    sd, _ = compile_results(Path("outputs/transcripts"), Path("outputs/results"))
    print(f"Compiled from transcripts: {len(sd)} rows", flush=True)
