import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import compute_model_metrics, inference


def compute_slices():
    """Compute metrics on slice of data (test set)"""

    data = pd.read_csv(r"../data/processed/census_processed.csv")


    _, test = train_test_split(data, test_size=0.20, random_state=42)


    with open("../model/model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("../model/encoder.pkl", "rb") as file:
        encoder = pickle.load(file)

    with open("../model/label_binarizer.pkl", "rb") as file:
        lb = pickle.load(file)

    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    with open("slice_output.txt", "w") as file:
        for col_value in test["education"].unique():
            file.write(f"Slice -> Column: {col_value}\n")

            X_test, y_test, encoder, lb = process_data(
                test.loc[test["education"] == col_value],
                categorical_features=categorical_features,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb,
            )

            preds = inference(model, X_test)

            precision, recall, fbeta = compute_model_metrics(y_test, preds)

            file.write(f"Precision:{precision:.4f}\n")
            file.write(f"Recall:{recall:.4f}\n")
            file.write(f"FBeta:{fbeta:.4f}\n")
            file.write("\n")


if __name__ == "__main__":
    compute_slices()