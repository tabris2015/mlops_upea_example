import logging
import argparse
from trainer.workflow import preprocess, train_DT, evaluate_model, export_model

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        help="Dataset file on GCS",
        type=str,
    )

    args = parser.parse_args()

    datasets = preprocess(
        dataset_path=args.data_path, 
        categorical_columns=["gender", "ever_married", "work_type", "Residence_type", "smoking_status"], 
        target_column="stroke",
    )

    model = train_DT(datasets["X_train"], datasets["y_train"])

    evaluate_model(datasets["X_test"], datasets["y_test"], model)

    export_model(model, "model.joblib")


if __name__ == "__main__":
    main()
