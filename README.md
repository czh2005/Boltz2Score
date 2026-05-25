# Boltz2Score Pipeline

A pipeline for evaluating biomolecular structure quality using Boltz2Score.

Boltz2Score is a score-only adaptation of Boltz-2 for structure evaluation. It reuses the Boltz-2 confidence head to score an existing input structure, skipping the structure-generation step when `score_only=True`.

## Environment Setup

### 1. Create and Activate Conda Environment
```bash
conda create -n boltz2score python=3.10
conda activate boltz2score
```

### 2. Install Boltz2Score and Dependencies
```bash
git clone https://github.com/czh2005/Boltz2Score.git
cd Boltz2Score/

# Install Python dependencies
pip install -e .
```

The first run downloads the Boltz-2 confidence checkpoint and required cache files into `~/.boltz` by default.

### 3. Input Files

Example FASTA files are stored in:

```text
examples/inputs/
```

Generated benchmark samples are written to:

```text
samples/
```

Prediction and score-only outputs are written to:

```text
outputs/results/
```

## Usage Pipeline

The **Boltz2Score pipeline** is designed for evaluating biomolecular structures with the Boltz-2 confidence module. It supports both full Boltz-2 prediction and score-only evaluation of structures generated in a previous run.

### 1. Single Sample Benchmark

`benchmark_1_sample.py` runs a full Boltz-2 prediction on `examples/inputs/1CRN.fasta`, then evaluates the generated structure in score-only mode.

```bash
python benchmark_1_sample.py
```

Outputs are saved under:

```text
outputs/results/benchmark_out_full_1/
outputs/results/benchmark_out_score_1/
```

### 2. Multi-chain Benchmark

`benchmark_20_multi_chain_new.py` creates 20 copies of the multi-chain FASTA example, runs full prediction, then runs score-only evaluation on each generated structure.

```bash
python benchmark_20_multi_chain_new.py
```

Generated FASTA files are saved under:

```text
samples/samples_20_multi_chain_new/
```

Outputs are saved under:

```text
outputs/results/benchmark_out_full_multi_20_new/
outputs/results/benchmark_out_score_multi_20_new/
```

### 3. Score-only Run

If full prediction results already exist, `run_score_only.py` reruns only the score-only stage for the 20 multi-chain samples.

```bash
python run_score_only.py
```

This script expects structures from:

```text
outputs/results/benchmark_out_full_multi_20_new/
```

and writes score-only outputs to:

```text
outputs/results/benchmark_out_score_multi_20_new/
```

### 4. Extract Metrics

After running the multi-chain benchmark, compare iPTM values from full Boltz-2 prediction and Boltz2Score:

```bash
python print_iptm.py
```

or:

```bash
python extract_iptm.py
```

## Output Metrics

The pipeline reads the Boltz confidence JSON files and reports the following metrics:

| Metric | Level | Description |
| --- | --- | --- |
| **pTM** | Global | Predicted TM-score for the overall structure. |
| **ipTM** | Inter-chain | Interface predicted TM-score for multi-chain interactions. |
| **pLDDT** | Per-residue / Per-chain | Predicted local confidence score. Higher values indicate higher local confidence. |
| **PAE** | Pairwise residue level | Predicted aligned error between residue pairs. Lower values indicate higher confidence in relative positioning. |

The primary benchmark scripts compare total runtime between full Boltz-2 prediction and score-only Boltz2Score evaluation.

## Repository Layout

```text
Boltz2Score/
|-- boltz_server.py              # Boltz2Server wrapper with score-only support
|-- boltzscore_paths.py          # Shared project paths for examples, samples, and outputs
|-- benchmark_*.py               # Full-prediction and score-only benchmark scripts
|-- run_score_only.py            # Score-only runner for existing full-prediction outputs
|-- print_iptm.py                # Metric comparison helper
|-- extract_iptm.py              # Compact iPTM extraction helper
|-- examples/
|   |-- inputs/                  # Boltz2Score FASTA examples
|   `-- boltz/                   # Upstream Boltz examples
|-- samples/                     # Generated benchmark FASTA files
|-- outputs/
|   |-- results/                 # Prediction and score-only outputs
|   `-- logs/                    # Benchmark logs
|-- src/boltz/                   # Boltz source package
|-- scripts/boltz/               # Upstream Boltz processing/training scripts
`-- docs/                        # Boltz docs and local analysis artifacts
```

## Reference

For more information about Boltz, please visit the [Boltz GitHub repository](https://github.com/jwohlwend/boltz).

For the AF3Score project style used as a reference for this repository layout, see [AF3Score](https://github.com/Mingchenchen/AF3Score).
