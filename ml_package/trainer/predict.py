import logging
import argparse
import pandas as pd
from trainer.workflow import preprocess, load_model

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        help="Model file path",
        type=str,
    )

    parser.add_argument(
        "--input_path",
        help="inputs csv file path",
        type=str,
    )

    args = parser.parse_args()

    model = load_model(args.model_path)
    to_predict = pd.read_csv(args.input_path)
    preds = model.predict(to_predict)
    print(preds)


if __name__ == "__main__":
    main()