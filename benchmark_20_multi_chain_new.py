import time
from pathlib import Path
from boltz_server import Boltz2Server
from boltzscore_paths import example_input, sample_dir, output_dir, prediction_cif, confidence_json, project_path

def setup_samples(num_samples=20):
    base_fasta = example_input("multi_chain.fasta")
    samples_path = sample_dir("samples_20_multi_chain_new")
    fasta_paths = []
    for i in range(num_samples):
        new_fasta = samples_path / f"sample_multi_{i}.fasta"
        with open(base_fasta, 'r') as f:
            content = f.read()
        
        with open(new_fasta, 'w') as f:
            f.write(content)
            
        fasta_paths.append(new_fasta)
        
    return fasta_paths

def run_benchmark():
    print("Setting up 20 multi-chain samples...")
    fasta_paths = setup_samples(20)
    
    print("\nInitializing Boltz2Server (loading model)...")
    server = Boltz2Server()
    
    # 1. Full Prediction for 20 samples
    print("\n--- Starting Full Prediction for 20 samples ---")
    full_start = time.time()
    for i, fasta in enumerate(fasta_paths):
        print(f"\n[Full Prediction] Processing sample {i+1}/20: {fasta.name}")
        server.predict(
            data_path=str(fasta),
            out_dir=output_dir("benchmark_out_full_multi_20_new"),
            score_only=False,
            use_msa_server=True
        )
    full_time = time.time() - full_start
    print(f"\nFull prediction for 20 samples took: {full_time:.2f} seconds")
    
    # 2. Score Only for 20 samples
    print("\n--- Starting Score Only for 20 samples ---")
    score_start = time.time()
    for i, fasta in enumerate(fasta_paths):
        print(f"\n[Score Only] Processing sample {i+1}/20: {fasta.name}")
        # We need the input structure from the full prediction
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
    
    print("\n==================================================")
    print("                    SUMMARY                       ")
    print("==================================================")
    print(f"20 samples Full Prediction (boltz2) : {full_time:.2f} seconds")
    print(f"20 samples Score Only (Boltz2score) : {score_time:.2f} seconds")
    print(f"Absolute Time Saved                 : {full_time - score_time:.2f} seconds")
    print(f"Relative Speedup                    : {full_time / score_time:.2f}x")
    print("==================================================")

if __name__ == "__main__":
    run_benchmark()
