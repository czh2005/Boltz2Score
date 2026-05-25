from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from boltz_server import Boltz2Server
from boltzscore_paths import example_input, sample_dir, output_dir, prediction_cif, confidence_json, project_path
server = Boltz2Server()
print("Trunk Pairformer blocks:", server.model_module.pairformer_module.num_blocks)
print("Confidence Pairformer blocks:", server.model_module.confidence_module.pairformer_stack.num_blocks)
print("Recycling steps:", server.predict_args["recycling_steps"])
