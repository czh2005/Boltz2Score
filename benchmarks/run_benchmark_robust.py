from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import json
import subprocess
import os
from boltzscore_paths import output_dir, prediction_cif, project_path

def run_robust_benchmark():
    fasta_dir = project_path("examples/inputs/test_cases_20_boltz")
    fastas = sorted(list(fasta_dir.glob("*.fasta")), key=lambda x: x.stat().st_size)
    
    results_file = project_path("outputs/results/benchmark_results.json")
    if results_file.exists():
        with open(results_file, "r") as f:
            results = json.load(f)
    else:
        results = {}

    print(f"Found {len(fastas)} fasta files.")
    
    # 1. Full Prediction
    for fasta in fastas:
        name = fasta.stem
        fasta_arg = str(fasta).replace("\\", "\\\\")
        if name not in results:
            results[name] = {}
            
        if "full_time" not in results[name]:
            print(f"\n[Full Prediction] Processing {name}...")
            start = time.time()
            
            # Run via subprocess to avoid crashing the main script on OOM
            cmd = [
                "python3", "-c",
                f"""
from boltz_server import Boltz2Server
from boltzscore_paths import example_input, sample_dir, output_dir, prediction_cif, confidence_json, project_path
server = Boltz2Server()
server.predict(
    data_path='{fasta_arg}',
    out_dir=output_dir('benchmark_out_full_specific_20'),
    score_only=False,
    use_msa_server=True
)
"""
            ]
            
            try:
                subprocess.run(cmd, check=True)
                full_time = time.time() - start
                results[name]["full_time"] = full_time
                print(f"Success! Took {full_time:.2f}s")
            except subprocess.CalledProcessError:
                print(f"Failed on {name}")
                results[name]["full_time"] = None
                
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

    # 2. Score Only
    for fasta in fastas:
        name = fasta.stem
        if "score_time" not in results[name]:
            print(f"\n[Score Only] Processing {name}...")
            
            input_struct = prediction_cif("benchmark_out_full_specific_20", name)
            input_struct_arg = str(input_struct).replace("\\", "\\\\")
            if not os.path.exists(input_struct):
                print(f"Input structure not found for {name}, skipping score_only.")
                results[name]["score_time"] = None
                continue
                
            start = time.time()
            
            cmd = [
                "python3", "-c",
                f"""
from boltz_server import Boltz2Server
from boltzscore_paths import example_input, sample_dir, output_dir, prediction_cif, confidence_json, project_path
server = Boltz2Server()
server.predict(
    data_path='{fasta_arg}',
    out_dir=output_dir('benchmark_out_score_specific_20'),
    score_only=True,
    input_structure='{input_struct_arg}',
    use_msa_server=True
)
"""
            ]
            
            try:
                subprocess.run(cmd, check=True)
                score_time = time.time() - start
                results[name]["score_time"] = score_time
                print(f"Success! Took {score_time:.2f}s")
            except subprocess.CalledProcessError:
                print(f"Failed on {name}")
                results[name]["score_time"] = None
                
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

    # Summary
    total_full = 0
    total_score = 0
    valid_count = 0
    
    for name, data in results.items():
        if data.get("full_time") is not None and data.get("score_time") is not None:
            total_full += data["full_time"]
            total_score += data["score_time"]
            valid_count += 1
            
    print("\n==================================================")
    print("                    SUMMARY                       ")
    print("==================================================")
    print(f"Valid samples compared              : {valid_count}/{len(fastas)}")
    print(f"Total Full Prediction (boltz2)      : {total_full:.2f} seconds")
    print(f"Total Score Only (Boltz2score)      : {total_score:.2f} seconds")
    print(f"Absolute Time Saved                 : {total_full - total_score:.2f} seconds")
    if total_score > 0:
        print(f"Relative Speedup                    : {total_full / total_score:.2f}x")
    print("==================================================")

if __name__ == "__main__":
    run_robust_benchmark()
