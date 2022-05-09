from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pickle
import pandas as pd

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# Add code to load in the data.
data = pd.read_csv(r"../data/processed/census_processed.csv", index_col=0)

# Optional enhancement, use K-fold cross validation
# instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process the train data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Train and save a model.
model = train_model(X_train, y_train)

# Save models
pickle.dump(model, open("../model/model.pkl", "wb"))
pickle.dump(encoder, open("../model/encoder.pkl", "wb"))
pickle.dump(lb, open("../model/label_binarizer.pkl", "wb"))

# Process the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

print(f'Precision: {precision}, Recall: {recall}, FBeta: {fbeta}')
