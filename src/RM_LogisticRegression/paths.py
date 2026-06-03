from datetime import datetime
from pathlib import Path

from RM_LogisticRegression.constants import METHOD_NAME


def run_output_dir(run_name: str | None = None) -> Path:
    run_folder = run_name or datetime.now().strftime("%Y-%m-%d_%H-%M")
    return Path("output") / run_folder / METHOD_NAME


def default_score_output_path(
    input_path: Path,
    limit: int | None,
    run_name: str | None = None,
) -> Path:
    suffix = f"_limit{limit}" if limit is not None else ""
    return run_output_dir(run_name) / f"{input_path.stem}_gemma_rm_scores{suffix}.csv"


def default_calibrator_output_path(run_name: str | None = None) -> Path:
    return run_output_dir(run_name) / "rm_calibrated_valid_predictions.csv"


def default_calibrator_model_path(run_name: str | None = None) -> Path:
    return run_output_dir(run_name) / "rm_logistic_regression_model.joblib"


def default_mlp_calibrator_output_path(run_name: str | None = None) -> Path:
    return run_output_dir(run_name) / "rm_mlp_calibrated_valid_predictions.csv"


def default_mlp_calibrator_model_path(run_name: str | None = None) -> Path:
    return run_output_dir(run_name) / "rm_mlp_calibrator.pt"


def default_mlp_search_results_path(run_name: str | None = None) -> Path:
    return run_output_dir(run_name) / "rm_mlp_hparam_search_results.csv"


def default_mlp_search_best_output_path(run_name: str | None = None) -> Path:
    return run_output_dir(run_name) / "rm_mlp_hparam_best_valid_predictions.csv"


def default_mlp_search_best_model_path(run_name: str | None = None) -> Path:
    return run_output_dir(run_name) / "rm_mlp_hparam_best_model.pt"
