from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
EXAMPLES_DIR = PROJECT_ROOT / "examples" / "inputs"
SAMPLES_DIR = PROJECT_ROOT / "samples"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "results"


def project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def example_input(name: str) -> Path:
    return EXAMPLES_DIR / name


def sample_dir(name: str) -> Path:
    path = SAMPLES_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_dir(name: str) -> Path:
    path = OUTPUTS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def boltz_result_dir(output_name: str, target_name: str) -> Path:
    return output_dir(output_name) / f"boltz_results_{target_name}"


def prediction_cif(output_name: str, target_name: str, model_idx: int = 0) -> Path:
    return (
        boltz_result_dir(output_name, target_name)
        / "predictions"
        / target_name
        / f"{target_name}_model_{model_idx}.cif"
    )


def confidence_json(output_name: str, target_name: str, model_idx: int = 0) -> Path:
    return (
        boltz_result_dir(output_name, target_name)
        / "predictions"
        / target_name
        / f"confidence_{target_name}_model_{model_idx}.json"
    )
