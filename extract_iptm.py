import json
import os
from boltzscore_paths import confidence_json

print(f"{'Sample':<15} | {'Boltz2 iptm':<15} | {'Boltz2-score iptm':<15}")
print("-" * 53)

for i in range(20):
    sample_name = f"sample_multi_{i}"
    
    # Boltz2
    boltz2_path = confidence_json("benchmark_out_full_multi_20_new", sample_name)
    boltz2_iptm = "N/A"
    if os.path.exists(boltz2_path):
        with open(boltz2_path, "r") as f:
            data = json.load(f)
            val = data.get("iptm", "N/A")
            if val != "N/A":
                boltz2_iptm = f"{val:.4f}"
            
    # Boltz2-score
    score_path = confidence_json("benchmark_out_score_multi_20_new", sample_name)
    score_iptm = "N/A"
    if os.path.exists(score_path):
        with open(score_path, "r") as f:
            data = json.load(f)
            val = data.get("iptm", "N/A")
            if val != "N/A":
                score_iptm = f"{val:.4f}"
            
    print(f"{sample_name:<15} | {boltz2_iptm:<15} | {score_iptm:<15}")
