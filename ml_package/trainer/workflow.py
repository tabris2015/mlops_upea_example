import os
import joblib
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve

LOGGER = logging.getLogger(__name__)

def preprocess(dataset_path: str, categorical_columns: list[str], target_column: str, test_size: float = 0.30) -> dict[str, np.ndarray]:
    LOGGER.info(f"Reading raw dataset from: {dataset_path}")
    dataset_df = pd.read_csv(dataset_path)
    
    # one hot encoding categorical columns
    dataset_df = pd.get_dummies(dataset_df, columns=categorical_columns)

    # fill null values
    dataset_df = dataset_df.fillna(dataset_df.mean())

    # split features and target
    X, y = dataset_df.drop(columns=["id", target_column]), dataset_df[target_column].values

    # split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}


def train_DT(X_train: np.ndarray, y_train:np.ndarray) -> DecisionTreeClassifier:
    LOGGER.info(f"training Decision Tree Model")
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    return model

def evaluate_model(X_test: np.ndarray, y_test: np.ndarray, model: DecisionTreeClassifier) -> dict:
    mean_acc = model.score(X_test, y_test)
    LOGGER.info(f"mean accuracy: {mean_acc}")
    return {"mean_acc": mean_acc}

def export_model(model: DecisionTreeClassifier, local_path: str):
    LOGGER.info(f"saving model locally as {local_path}")
    joblib.dump(model, local_path)

    if os.environ.get("AIP_MODEL_DIR"):
        LOGGER.info(f"uploading model to GCP")
        model_directory = os.environ["AIP_MODEL_DIR"]
        storage_path = os.path.join(model_directory, local_path)
        blob = storage.Blob.from_string(storage_path, client=storage.Client())
        blob.upload_from_filename(local_path)
        logging.info(f"model exported to: {storage_path}")

def load_model(local_path: str):
    LOGGER.info(f"loading model locally from {local_path}")
    model = joblib.load(local_path)

    return model