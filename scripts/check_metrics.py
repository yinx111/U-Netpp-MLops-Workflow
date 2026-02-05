import argparse
import json
import os
import sys

from env_loader import load_dotenv

load_dotenv()


def load_metrics(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"metrics file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Quality gate for metrics.json")
    parser.add_argument("--metrics", default="outputs/metrics.json", help="Path to metrics.json")
    parser.add_argument("--miou", type=float, default=None, help="Minimum test mIoU")
    parser.add_argument("--oa", type=float, default=None, help="Minimum test OA")
    parser.add_argument(
        "--config",
        default="configs/train.yaml",
        help="Optional config to read gate thresholds",
    )
    parser.add_argument("--stamp", default=None, help="Optional path to write a pass stamp file")
    args = parser.parse_args()

    min_miou = args.miou
    min_oa = args.oa
    if (min_miou is None or min_oa is None) and args.config and os.path.exists(args.config):
        try:
            import yaml

            with open(args.config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            gate_cfg = cfg.get("gate", {}) if isinstance(cfg, dict) else {}
            if min_miou is None and gate_cfg.get("min_miou") is not None:
                min_miou = float(gate_cfg.get("min_miou"))
            if min_oa is None and gate_cfg.get("min_oa") is not None:
                min_oa = float(gate_cfg.get("min_oa"))
        except Exception:
            pass
    if min_miou is None:
        env = os.getenv("MIN_MIOU")
        min_miou = float(env) if env is not None else 0.0
    if min_oa is None:
        env = os.getenv("MIN_OA")
        min_oa = float(env) if env is not None else 0.0

    data = load_metrics(args.metrics)
    test_miou = data.get("test_mIoU")
    test_oa = data.get("test_OA")

    missing = [k for k in ("test_mIoU", "test_OA") if data.get(k) is None]
    if missing:
        print(f"[Gate] Missing metrics keys: {missing}")
        return 2

    print(f"[Gate] test_mIoU={test_miou} (min {min_miou})")
    print(f"[Gate] test_OA={test_oa} (min {min_oa})")

    if test_miou < min_miou or test_oa < min_oa:
        print("[Gate] FAILED: metrics below threshold")
        return 1

    if args.stamp:
        os.makedirs(os.path.dirname(args.stamp) or ".", exist_ok=True)
        with open(args.stamp, "w", encoding="utf-8") as f:
            f.write("ok\n")

    print("[Gate] PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
