from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from boltzscore_paths import confidence_json

def print_iptm_values():
    print(f"{'Sample':<15} | {'Full Prediction iPTM':<25} | {'Score Only iPTM':<25}")
    print("-" * 70)
    
    for i in range(20):
        sample_name = f"sample_multi_{i}"
        
        # Full prediction path
        full_json_path = confidence_json("benchmark_out_full_multi_20_new", sample_name)
        
        # Score only path
        score_json_path = confidence_json("benchmark_out_score_multi_20_new", sample_name)
        
        full_iptm = "N/A"
        if full_json_path.exists():
            with open(full_json_path, 'r') as f:
                data = json.load(f)
                val = data.get('iptm')
                full_iptm = f"{val:.4f}" if val is not None else "N/A"
                
        score_iptm = "N/A"
        if score_json_path.exists():
            with open(score_json_path, 'r') as f:
                data = json.load(f)
                val = data.get('iptm')
                score_iptm = f"{val:.4f}" if val is not None else "N/A"
                
        print(f"{sample_name:<15} | {full_iptm:<25} | {score_iptm:<25}")

if __name__ == "__main__":
    print_iptm_values()
