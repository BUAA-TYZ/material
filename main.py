# main.py
import argparse
from src.pipeline import predict_external_xlsx, run_pipeline
from src.config import DEFAULT_MODEL_NAME


def parse_args():
    parser = argparse.ArgumentParser(description="Materials ML pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model name: rf | gbrt | xgb | svr",
    )
    parser.add_argument(
        "--predict-xlsx",
        type=str,
        default=None,
        help="Path to an external xlsx file for prediction.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output xlsx path for prediction results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.predict_xlsx:
        predict_external_xlsx(args.predict_xlsx, args.model, args.output)
    else:
        run_pipeline(args.model)


if __name__ == "__main__":
    main()
