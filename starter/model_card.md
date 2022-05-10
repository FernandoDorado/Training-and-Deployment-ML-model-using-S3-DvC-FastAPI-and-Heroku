# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

We use a Random Forest Classifier without changes for this classification task. It is a p-o-c, the hyperparameters could be improved. 

## Intended Use

Our main purpose is to predict whether a person's income is above or below $50,000 per year from census data. In addition, we intend to apply recently acquired knowledge to develop an API using FastAPI to serve an ML model.

## Training Data
The description of the dataset can be found at: https://archive.ics.uci.edu/ml/datasets/census+income. For training, 80% of the original dataset was randomly selected.

## Evaluation Data
The model has been tested using the 20% of the data. 

## Metrics
Precision, recall and F-beta metrics were chosen to evaluate the model's performance:

- Precision: 0.7761075949367089
- Recall: 0.6244430299172502
- F-beta: 0.692063492063492

## Ethical Considerations
This dataset was taken from the UCI repository and is already anonymised. However, the inherent bias needs to be carefully analysed.


## Caveats and Recommendations
The dataset was obtained from the 1994 census database.

