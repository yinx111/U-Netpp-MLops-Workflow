import argparse
import json
import os

import mlflow
import yaml

from env_loader import load_dotenv

load_dotenv()


def main() -> int:
    parser = argparse.ArgumentParser(description="Register MLflow model from a run")
    parser.add_argument("--run-id", default=None, help="MLflow run id")
    parser.add_argument("--model-name", default=None, help="Registered model name")
    parser.add_argument(
        "--model-uri",
        default=None,
        help="Direct MLflow model URI (overrides run-id/artifact-path if provided)",
    )
    parser.add_argument(
        "--artifact-path",
        default="model",
        help="Artifact path used when logging the model (default: model)",
    )
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI")
    parser.add_argument("--stage", default=None, help="Optional stage to transition to")
    parser.add_argument(
        "--metrics",
        default="outputs/metrics.json",
        help="Optional metrics.json to read mlflow_run_id",
    )
    parser.add_argument(
        "--config",
        default="configs/train.yaml",
        help="Optional config to read mlflow settings",
    )
    parser.add_argument(
        "--stamp",
        default=None,
        help="Optional path to write a stamp file when done",
    )
    args = parser.parse_args()

    run_id = args.run_id or os.getenv("MLFLOW_RUN_ID")
    model_name = args.model_name or os.getenv("MLFLOW_MODEL_NAME")
    model_uri = args.model_uri or os.getenv("MLFLOW_MODEL_URI")
    tracking_uri = args.tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    stage = args.stage or os.getenv("MLFLOW_MODEL_STAGE")

    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            mlflow_cfg = cfg.get("mlflow", {}) if isinstance(cfg, dict) else {}
            if not model_name:
                model_name = mlflow_cfg.get("model_name")
            if not stage:
                stage = mlflow_cfg.get("model_stage")
            if not tracking_uri:
                tracking_uri = mlflow_cfg.get("tracking_uri")
        except Exception:
            pass

    if args.metrics and os.path.exists(args.metrics):
        try:
            with open(args.metrics, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            if not model_uri:
                model_uri = metrics_data.get("mlflow_model_uri")
            if not run_id:
                run_id = metrics_data.get("mlflow_run_id")
        except Exception:
            run_id = run_id

    if not (model_uri or run_id) or not model_name:
        print("[Register] Skipped: model uri/run id or model name not provided")
        if args.stamp:
            os.makedirs(os.path.dirname(args.stamp) or ".", exist_ok=True)
            with open(args.stamp, "w", encoding="utf-8") as f:
                f.write("skipped\n")
        return 0

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if not model_uri:
        model_uri = f"runs:/{run_id}/{args.artifact_path}"

    result = mlflow.register_model(model_uri, model_name)
    print(f"[Register] name={result.name} version={result.version}")

    if stage:
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=result.name, version=result.version, stage=stage
        )
        print(f"[Register] transitioned to stage: {stage}")

    if args.stamp:
        os.makedirs(os.path.dirname(args.stamp) or ".", exist_ok=True)
        with open(args.stamp, "w", encoding="utf-8") as f:
            f.write(f"{result.name}:{result.version}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
