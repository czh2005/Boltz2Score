from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import json
from boltz_server import Boltz2Server
from boltzscore_paths import example_input, sample_dir, output_dir, prediction_cif, confidence_json, project_path

def run_one():
    print("Initializing Boltz2Server...")
    server = Boltz2Server()
    
    fasta = sample_dir("samples_20_multi_chain_new") / "sample_multi_0.fasta"
    input_struct = prediction_cif("benchmark_out_full_multi_20_new", fasta.stem)
    out_dir = output_dir("out_score_one_sample")
    
    print(f"Running score_only for {fasta.name}...")
    server.predict(
        data_path=str(fasta),
        out_dir=out_dir,
        score_only=True,
        input_structure=input_struct,
        use_msa_server=True
    )
    
    json_path = confidence_json("out_score_one_sample", fasta.stem)
    if json_path.exists():
        with open(json_path, 'r') as f:
            data = json.load(f)
            print(f"\nResults for {fasta.name}:")
            print(f"iptm: {data.get('iptm', 'N/A')}")
            print(f"ptm: {data.get('ptm', 'N/A')}")
    else:
        print("Output JSON not found.")

if __name__ == "__main__":
    run_one()
