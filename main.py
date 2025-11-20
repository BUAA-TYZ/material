# main.py
import argparse
from src.pipeline import run_pipeline
from src.config import DEFAULT_MODEL_NAME


def parse_args():
    parser = argparse.ArgumentParser(description="Materials ML pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model name: rf | gbrt | xgb | svr",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_pipeline(args.model)


if __name__ == "__main__":
    main()