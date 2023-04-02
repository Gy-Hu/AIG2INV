import pandas as pd
import numpy as np
import sympy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse
from pprint import pprint

import sklearn.datasets
import sklearn.metrics
import autosklearn.regression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import autosklearn.pipeline.components.feature_preprocessing
import autosklearn.classification
poly = PolynomialFeatures(degree=2, include_bias=False)

def parse_table(path):
    # parse the csv table to pandas dataframe
    with open(path, 'r') as f:
        df = pd.read_csv(f)
        
        # only keep the row that  res is "unsat"
        df = df[df["res"] == "unsat"]
        print(df)
        return df

if __name__ == '__main__':
    # add parser
    parser = argparse.ArgumentParser(description="Convert aig to aag automatically")
    #parser.add_argument('--model-file', type=str, default=None, help='The path to the model file')
    parser.add_argument('--validate', action='store_true', help='Determin whether to validate the model')
    #parser.add_argument('--model', type=int, default=1, help='Determin which model to use')
    args = parser.parse_args()
    '''
    --------------------Get the aig list (and their path)-------------------
    '''
    # read table at first    
    df = parse_table("/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/clause-learning/data-collect/stat/size20.csv")
    # prepare data
    #X = 2 * np.random.randn(100, 5)
    
    # convert the dataframe to numpy array, X = [M, I, L, O, A]
    df = df.drop(columns=["O"])
    X = df.iloc[:, 1:5].to_numpy()
    
    # y = [n_clauses]
    y = df.iloc[:, -3].to_numpy() # -3 -> time, -1 -> n_clauses
    
    '''
    -----------------------------------Feature Engineering---------------------------
    '''
    # Apply feature engineering (e.g., create polynomial features)
    X_poly = poly.fit_transform(X)
    X_interaction = np.column_stack((X, X[:, 0] * X[:, 1], X[:, 0] * X[:, 2], X[:, 0] * X[:, 3], X[:, 1] * X[:, 2], X[:, 1] * X[:, 3], X[:, 2] * X[:, 3]))
    X_ratio = X_ratio = np.column_stack((X, X[:, 0] / X[:, 1], X[:, 2] / X[:, 3]))
    
    # Replace X with the new feature matrix, e.g., X_poly, X_interaction, or X_ratio
    X = X_poly

    # Standardize the new features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # logarithmic transformation
    # y = np.log1p(y)
    '''
    ------------------------------------Feature Selection-----------------------------
    '''
    # Create a dictionary of feature preprocessors
    #feature_preprocessors = autosklearn.pipeline.components.feature_preprocessing.get_components()

    # Print the available feature preprocessors
    #for name, obj in feature_preprocessors.items():
    #    print(name, obj)
    
    '''
    -------------------------------------Model Selection and train-----------------------------
    '''
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # select model 
    automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=600,
    per_run_time_limit=90,
    ensemble_size=60,
    tmp_folder="/data/guangyuh/tmp/autosklearn_regression_example_tmp"
    # include={
    # 'feature_preprocessor': [
    #     'select_percentile_regression',
    #     'select_rates_regression'
    # ]}
    )
    automl.fit(X_train, y_train, dataset_name='aig_time_predict')
    
    # View the models found by auto-sklearn
    print(automl.leaderboard())
    
    # Evaluate model performance on test data
    train_predictions = automl.predict(X_train)
    print("Train score:", sklearn.metrics.r2_score(y_train, train_predictions))
    test_predictions = automl.predict(X_test)
    print("Test score:", sklearn.metrics.r2_score(y_test, test_predictions))
    mae = sklearn.metrics.mean_absolute_error(y_test, test_predictions)
    print("MAE: %.3f" % mae)
    
    # Plot the predictions and save the plot
    #if args.validate:
    plt.scatter(train_predictions, y_train, label="Train samples", c="#ff7f0e")
    plt.scatter(test_predictions, y_test, label="Test samples", c="#1f77b4")
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.legend()
    plt.plot([30, 4000], [30, 4000], c="k", zorder=0)
    plt.xlim([30, 4000])
    plt.ylim([30, 4000])
    plt.tight_layout()
    plt.savefig("./predictions_autosklearn.png")
    