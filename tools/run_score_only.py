from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
from boltz_server import Boltz2Server
from boltzscore_paths import example_input, sample_dir, output_dir, prediction_cif, confidence_json, project_path

def run_score_only():
    print("\nInitializing Boltz2Server (loading model)...")
    server = Boltz2Server()
    
    fasta_paths = [sample_dir("samples_20_multi_chain_new") / f"sample_multi_{i}.fasta" for i in range(20)]
    
    print("\n--- Starting Score Only for 20 samples ---")
    score_start = time.time()
    for i, fasta in enumerate(fasta_paths):
        print(f"\n[Score Only] Processing sample {i+1}/20: {fasta.name}")
        input_struct = prediction_cif("benchmark_out_full_multi_20_new", fasta.stem)
        server.predict(
            data_path=str(fasta),
            out_dir=output_dir("benchmark_out_score_multi_20_new"),
            score_only=True,
            input_structure=input_struct,
            use_msa_server=True
        )
    score_time = time.time() - score_start
    print(f"\nScore only for 20 samples took: {score_time:.2f} seconds")

if __name__ == "__main__":
    run_score_only()
