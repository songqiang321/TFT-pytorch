"""Default data formatting functions for experiments.

For new datasets, inherit form GenericDataFormatter and implement
all abstract functions.

These dataset-specific methods:
1) Define the column and input types for tabular dataframes used by model
2) Perform the necessary input feature engineering & normalisation steps
3) Reverts the normalisation for predictions
4) Are responsible for train, validation and test splits


"""
