import time
from pathlib import Path
from boltz_server import Boltz2Server
from boltzscore_paths import example_input, sample_dir, output_dir, prediction_cif, confidence_json, project_path

def run_single_benchmark():
    print("Initializing Boltz2Server (loading model)...")
    load_start = time.time()
    server = Boltz2Server()
    load_time = time.time() - load_start
    print(f"Model loading took: {load_time:.2f} seconds")
    
    fasta_path = example_input("1CRN.fasta")
    
    # 1. Full Prediction (Boltz2)
    print("\n--- Starting Full Prediction (Boltz2) ---")
    full_start = time.time()
    server.predict(
        data_path=fasta_path,
        out_dir=output_dir("benchmark_single_full"),
        score_only=False,
        use_msa_server=True
    )
    full_time = time.time() - full_start
    print(f"Full prediction took: {full_time:.2f} seconds")
    
    # 2. Score Only (Boltz2-score)
    print("\n--- Starting Score Only (Boltz2-score) ---")
    score_start = time.time()
    input_struct = prediction_cif("benchmark_single_full", "1CRN")
    server.predict(
        data_path=fasta_path,
        out_dir=output_dir("benchmark_single_score"),
        score_only=True,
        input_structure=input_struct,
        use_msa_server=False
    )
    score_time = time.time() - score_start
    print(f"Score only took: {score_time:.2f} seconds")
    
    print("\n==================================================")
    print("                    SUMMARY                       ")
    print("==================================================")
    print(f"Model Loading Time                  : {load_time:.2f} seconds")
    print(f"Full Prediction (Boltz2) Time       : {full_time:.2f} seconds")
    print(f"Score Only (Boltz2-score) Time      : {score_time:.2f} seconds")
    print(f"Absolute Time Saved                 : {full_time - score_time:.2f} seconds")
    print(f"Relative Speedup                    : {full_time / score_time:.2f}x")
    print("==================================================")

if __name__ == "__main__":
    run_single_benchmark()
