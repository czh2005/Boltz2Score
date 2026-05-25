import os
from pathlib import Path
from dataclasses import asdict
import torch
from pytorch_lightning import Trainer
from boltzscore_paths import example_input, output_dir, project_path

from boltz.main import (
    check_inputs,
    process_inputs,
    filter_inputs_structure,
    BoltzProcessedInput,
    download_boltz2,
    PairformerArgsV2,
    MSAModuleArgs,
    Boltz2DiffusionParams,
    BoltzSteeringParams,
)
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.model.models.boltz2 import Boltz2
from boltz.data.types import Manifest
from boltz.data.write.writer import BoltzWriter



class Boltz2Server:
    def __init__(self, cache_dir="~/.boltz", accelerator="gpu", devices=1):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.accelerator = accelerator
        self.devices = devices
        
        # Download model if needed
        download_boltz2(self.cache_dir)
        self.checkpoint = self.cache_dir / "boltz2_conf.ckpt"
        
        # Set up model parameters
        diffusion_params = Boltz2DiffusionParams()
        diffusion_params.step_scale = 1.5
        pairformer_args = PairformerArgsV2()
        pairformer_args.num_blocks = 4
        
        msa_args = MSAModuleArgs(
            subsample_msa=True,
            num_subsampled_msa=1024,
            use_paired_feature=True,
        )
        
        self.predict_args = {
            "recycling_steps": 1,
            "sampling_steps": 50,
            "diffusion_samples": 1,
            "max_parallel_samples": None,
            "write_confidence_summary": True,
            "write_full_pae": False,
            "write_full_pde": False,
        }
        
        steering_args = BoltzSteeringParams()
        steering_args.fk_steering = False
        steering_args.physical_guidance_update = False
        
        print("Loading Boltz2 model into memory...")
        self.model_module = Boltz2.load_from_checkpoint(
            self.checkpoint,
            strict=False,
            predict_args=self.predict_args,
            map_location="cpu",
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            use_kernels=False,
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
            steering_args=asdict(steering_args),
        )
        
        # Override confidence pairformer blocks
        self.model_module.confidence_module.pairformer_stack.num_blocks = 1
        self.model_module.confidence_module.pairformer_stack.layers = self.model_module.confidence_module.pairformer_stack.layers[:1]
        self.model_module.eval()
        print("Model loaded successfully!")

    def predict(self, data_path, out_dir, score_only=False, input_structure=None, use_msa_server=True, msa_dir=None):
        data = project_path(data_path)
        out_dir = project_path(out_dir)
        if input_structure is not None:
            input_structure = str(project_path(input_structure))
        out_dir = out_dir / f"boltz_results_{data.stem}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        data = check_inputs(data)
        
        ccd_path = self.cache_dir / "ccd.pkl"
        mol_dir = self.cache_dir / "mols"
        
        print(f"Processing inputs for {data}...")
        process_inputs(
            data=data,
            out_dir=out_dir,
            ccd_path=ccd_path,
            mol_dir=mol_dir,
            use_msa_server=use_msa_server,
            msa_server_url="https://api.colabfold.com",
            msa_pairing_strategy="greedy",
            msa_server_username=None,
            msa_server_password=None,
            api_key_header=None,
            api_key_value=None,
            boltz2=True,
            preprocessing_threads=1,
            max_msa_seqs=8192,
        )
        
        manifest = Manifest.load(out_dir / "processed" / "manifest.json")
        filtered_manifest = filter_inputs_structure(
            manifest=manifest,
            outdir=out_dir,
            override=True,
        )
        
        processed_dir = out_dir / "processed"
        processed = BoltzProcessedInput(
            manifest=filtered_manifest,
            targets_dir=processed_dir / "structures",
            msa_dir=processed_dir / "msa",
            constraints_dir=(processed_dir / "constraints") if (processed_dir / "constraints").exists() else None,
            template_dir=(processed_dir / "templates") if (processed_dir / "templates").exists() else None,
            extra_mols_dir=(processed_dir / "mols") if (processed_dir / "mols").exists() else None,
        )
        
        pred_writer = BoltzWriter(
            data_dir=processed.targets_dir,
            output_dir=out_dir / "predictions",
            output_format="mmcif",
            boltz2=True,
            write_embeddings=False,
        )
        
        trainer = Trainer(
            default_root_dir=out_dir,
            strategy="auto",
            callbacks=[pred_writer],
            accelerator=self.accelerator,
            devices=[0],
            precision="bf16-mixed",
        )
        
        data_module = Boltz2InferenceDataModule(
            manifest=processed.manifest,
            target_dir=processed.targets_dir,
            msa_dir=processed.msa_dir,
            mol_dir=mol_dir,
            num_workers=0, # Set to 0 to avoid multiprocessing issues in script
            constraints_dir=processed.constraints_dir,
            template_dir=processed.template_dir,
            extra_mols_dir=processed.extra_mols_dir,
            override_method=None,
            input_structure=input_structure if score_only else None,
        )
        
        if score_only:
            self.model_module.skip_run_structure = True
        else:
            self.model_module.skip_run_structure = False
            
        print(f"Running prediction for {data}...")
        trainer.predict(
            self.model_module,
            datamodule=data_module,
            return_predictions=False,
        )
        print(f"Prediction finished. Results saved to {out_dir}")

if __name__ == "__main__":
    import time
    
    # Initialize server (loads model once)
    server = Boltz2Server()
    
    # Example 1: Full prediction
    print("\n--- Starting Full Prediction ---")
    start_time = time.time()
    server.predict(
        data_path=example_input("1CRN.fasta"),
        out_dir=output_dir("server_out_1crn"),
        score_only=False
    )
    print(f"Full prediction took {time.time() - start_time:.2f} seconds")
    
    # Example 2: Score only
    print("\n--- Starting Score Only ---")
    start_time = time.time()
    server.predict(
        data_path=example_input("1CRN.fasta"),
        out_dir=output_dir("server_score_out_1crn"),
        score_only=True,
        input_structure=output_dir("server_out_1crn") / "boltz_results_1CRN/predictions/1CRN/1CRN_model_0.cif"
    )
    print(f"Score only took {time.time() - start_time:.2f} seconds")
