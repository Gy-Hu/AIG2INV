"""
======================
Tabular Regression
======================

The following example shows how to fit a sample regression model
with AutoPyTorch
"""
import os
import tempfile as tmp
import warnings
import pandas as pd
import argparse

import sklearn.datasets
import sklearn.model_selection

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from autoPyTorch.api.tabular_regression import TabularRegressionTask


############################################################################
# Data Loading
# ============
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
    df_2020 = parse_table("/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/clause-learning/data-collect/stat/size20.csv")
    df_2007 = parse_table("/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/clause-learning/data-collect/stat/size07.csv")
    
    # merge two dataframe
    df = pd.concat([df_2020, df_2007])
    
    # prepare data
    #X = 2 * np.random.randn(100, 5)
    
    # convert the dataframe to numpy array, X = [M, I, L, O, A]
    df = df.drop(columns=["O"])
    X = df.iloc[:, 1:5].to_numpy()
    
    # y = [n_clauses]
    y = df.iloc[:, -3].to_numpy() # -3 -> time, -1 -> n_clauses
    
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X,
        y,
        random_state=1,
    )

    ############################################################################
    # Build and fit a regressor
    # ==========================
    api = TabularRegressionTask()

    ############################################################################
    # Search for an ensemble of machine learning algorithms
    # =====================================================
    api.search(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test.copy(),
        y_test=y_test.copy(),
        optimize_metric='r2',
        total_walltime_limit=300,
        func_eval_time_limit_secs=50,
        memory_limit=40960,
        dataset_name="aig_checking_time_predict"
    )

    ############################################################################
    # Print the final ensemble performance before refit
    # =================================================
    
    y_pred = api.predict(X_test)
    score = api.score(y_pred, y_test)
    print(score)

    # Print statistics from search
    print(api.sprint_statistics())

    ###########################################################################
    # Refit the models on the full dataset.
    # =====================================

    api.refit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        dataset_name="aig_checking_time_predict",
        total_walltime_limit=500,
        memory_limit=40960,
        run_time_limit_secs=50
        # you can change the resampling strategy to
        # for example, CrossValTypes.k_fold_cross_validation
        # to fit k fold models and have a voting classifier
        # resampling_strategy=CrossValTypes.k_fold_cross_validation
    )

    ############################################################################
    # Print the final ensemble performance after refit
    # ================================================

    y_pred = api.predict(X_test)
    score = api.score(y_pred, y_test)
    print(score)
    
    ############################################################################
    # Print the predictions of the final ensemble and the true values
    # ===============================================================
    print(api.predict(X_test))
    print(y_test)

    # Print the final ensemble built by AutoPyTorch
    print(api.show_models())