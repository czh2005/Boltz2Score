from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
from boltz_server import Boltz2Server
from boltzscore_paths import example_input, sample_dir, output_dir, prediction_cif, confidence_json, project_path

def run_benchmark():
    fasta_path = example_input("multi_chain.fasta")
    
    print("\nInitializing Boltz2Server (loading model)...")
    server = Boltz2Server()
    
    # 1. Full Prediction for 1 sample
    print("\n--- Starting Full Prediction for 1 sample ---")
    full_start = time.time()
    server.predict(
        data_path=str(fasta_path),
        out_dir=output_dir("benchmark_out_full_multi"),
        score_only=False,
        use_msa_server=True # Enable MSA server
    )
    full_time = time.time() - full_start
    print(f"\nFull prediction took: {full_time:.2f} seconds")
    
    # 2. Score Only for 1 sample
    print("\n--- Starting Score Only for 1 sample ---")
    score_start = time.time()
    input_struct = prediction_cif("benchmark_out_full_multi", "multi_chain")
    server.predict(
        data_path=str(fasta_path),
        out_dir=output_dir("benchmark_out_score_multi"),
        score_only=True,
        input_structure=input_struct,
        use_msa_server=True
    )
    score_time = time.time() - score_start
    print(f"\nScore only took: {score_time:.2f} seconds")
    
    print("\n==================================================")
    print("                    SUMMARY                       ")
    print("==================================================")
    print(f"Full Prediction (boltz2) : {full_time:.2f} seconds")
    print(f"Score Only (Boltz2score) : {score_time:.2f} seconds")
    print(f"Absolute Time Saved      : {full_time - score_time:.2f} seconds")
    print(f"Relative Speedup         : {full_time / score_time:.2f}x")
    print("==================================================")

if __name__ == "__main__":
    run_benchmark()
