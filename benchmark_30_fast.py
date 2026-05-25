import time
from pathlib import Path
from boltz_server import Boltz2Server
from boltzscore_paths import example_input, sample_dir, output_dir, prediction_cif, confidence_json, project_path

def setup_samples(num_samples=30):
    base_fasta = example_input("1CRN.fasta")
    samples_path = sample_dir("samples_30")
    fasta_paths = []
    for i in range(num_samples):
        new_fasta = samples_path / f"sample_{i}.fasta"
        with open(base_fasta, 'r') as f:
            content = f.read()
        content = content.replace(">1CRN", f">sample_{i}")
        with open(new_fasta, 'w') as f:
            f.write(content)
        fasta_paths.append(new_fasta)
        
    return fasta_paths

def run_benchmark():
    print("Setting up 30 samples...")
    fasta_paths = setup_samples(30)
    
    print("\nInitializing Boltz2Server (loading model)...")
    server = Boltz2Server()
    
    # 1. Full Prediction for 30 samples
    print("\n--- Starting Full Prediction for 30 samples ---")
    full_start = time.time()
    for i, fasta in enumerate(fasta_paths):
        print(f"\n[Full Prediction] Processing sample {i+1}/30: {fasta.name}")
        try:
            server.predict(
                data_path=str(fasta),
                out_dir=output_dir("benchmark_out_full_30"),
                score_only=False,
                use_msa_server=True
            )
        except Exception as e:
            print(f"Error processing {fasta.name}: {e}")
    full_time = time.time() - full_start
    print(f"\nFull prediction for 30 samples took: {full_time:.2f} seconds")
    
    # 2. Score Only for 30 samples
    print("\n--- Starting Score Only for 30 samples ---")
    score_start = time.time()
    for i, fasta in enumerate(fasta_paths):
        print(f"\n[Score Only] Processing sample {i+1}/30: {fasta.name}")
        input_struct = prediction_cif("benchmark_out_full_30", fasta.stem)
        try:
            server.predict(
                data_path=str(fasta),
                out_dir=output_dir("benchmark_out_score_30"),
                score_only=True,
                input_structure=input_struct,
                use_msa_server=False
            )
        except Exception as e:
            print(f"Error processing {fasta.name}: {e}")
    score_time = time.time() - score_start
    print(f"\nScore only for 30 samples took: {score_time:.2f} seconds")
    
    print("\n==================================================")
    print("                    SUMMARY                       ")
    print("==================================================")
    print(f"30 samples Full Prediction (boltz2) : {full_time:.2f} seconds")
    print(f"30 samples Score Only (Boltz2score) : {score_time:.2f} seconds")
    print(f"Absolute Time Saved                 : {full_time - score_time:.2f} seconds")
    print(f"Relative Speedup                    : {full_time / score_time:.2f}x")
    print("==================================================")

if __name__ == "__main__":
    run_benchmark()
