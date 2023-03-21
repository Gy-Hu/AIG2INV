# /data/hongcezh/clause-learning/data-collect/stat/size20.csv
from pysr import PySRRegressor
import pandas as pd
import numpy as np
import sympy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def parse_table(path):
    # parse the csv table to pandas dataframe
    with open(path, 'r') as f:
        df = pd.read_csv(f)
        
        # only keep the row that  res is "unsat"
        df = df[df["res"] == "unsat"]
        print(df)
        return df

'''
# ---------------------------------------------------Model 1-------------------------------------------------
model = PySRRegressor(
    procs=4,
    populations=8,
    # ^ 2 populations per core, so one is always running.
    population_size=50,
    # ^ Slightly larger populations, for greater diversity.
    ncyclesperiteration=500, 
    # ^ Generations between migrations.
    niterations=10000000,  # Run forever
    early_stop_condition=(
        "stop_if(loss, complexity) = loss < 1e-6 && complexity < 10"
        # Stop early if we find a good and simple equation
    ),
    timeout_in_seconds=60 * 60 * 24,
    # ^ Alternatively, stop after 24 hours have passed.
    maxsize=50,
    # ^ Allow greater complexity.
    maxdepth=10,
    # ^ But, avoid deep nesting.
    binary_operators=["*", "+", "-", "/"],
    unary_operators=["square", "cube", "exp", "cos2(x)=cos(x)^2"],
    constraints={
        "/": (-1, 9),
        "square": 9,
        "cube": 9,
        "exp": 9,
    },
    # ^ Limit the complexity within each argument.
    # "inv": (-1, 9) states that the numerator has no constraint,
    # but the denominator has a max complexity of 9.
    # "exp": 9 simply states that `exp` can only have
    # an expression of complexity 9 as input.
    nested_constraints={
        "square": {"square": 1, "cube": 1, "exp": 0},
        "cube": {"square": 1, "cube": 1, "exp": 0},
        "exp": {"square": 1, "cube": 1, "exp": 0},
    },
    # ^ Nesting constraints on operators. For example,
    # "square(exp(x))" is not allowed, since "square": {"exp": 0}.
    complexity_of_operators={"/": 2, "exp": 3},
    # ^ Custom complexity of particular operators.
    complexity_of_constants=2,
    # ^ Punish constants more than variables
    select_k_features=4,
    # ^ Train on only the 4 most important features
    progress=True,
    # ^ Can set to false if printing to a file.
    weight_randomize=0.1,
    # ^ Randomize the tree much more frequently
    cluster_manager=None,
    # ^ Can be set to, e.g., "slurm", to run a slurm
    # cluster. Just launch one script from the head node.
    precision=64,
    # ^ Higher precision calculations.
    warm_start=True,
    # ^ Start from where left off.
    turbo=True,
    # ^ Faster evaluation (experimental)
    julia_project=None,
    # ^ Can set to the path of a folder containing the
    # "SymbolicRegression.jl" repo, for custom modifications.
    update=False,
    # ^ Don't update Julia packages
    extra_sympy_mappings={"cos2": lambda x: sympy.cos(x)**2},
    # extra_torch_mappings={sympy.cos: torch.cos},
    # ^ Not needed as cos already defined, but this
    # is how you define custom torch operators.
    # extra_jax_mappings={sympy.cos: "jnp.cos"},
    # ^ For JAX, one passes a string.
    )
# ---------------------------------------------------Model 1-------------------------------------------------
'''

'''
# ---------------------------------------------------Model 2-------------------------------------------------
'''
model = PySRRegressor(
    procs=4,
    populations=8,
    population_size=50,
    ncyclesperiteration=500,
    niterations=100000,
    early_stop_condition=(
        "stop_if(loss, complexity) = loss < 1e-6 && complexity < 10"
    ),
    timeout_in_seconds=60 * 60 * 24,
    maxsize=50,
    maxdepth=10,
    binary_operators=["*", "+", "-", "/"],
    unary_operators=["square", "cube", "exp", "cos2(x)=cos(x)^2"],
    constraints={
        "/": (-1, 9),
        "square": 9,
        "cube": 9,
        "exp": 9,
    },
    nested_constraints={
        "square": {"square": 1, "cube": 1, "exp": 0},
        "cube": {"square": 1, "cube": 1, "exp": 0},
        "exp": {"square": 1, "cube": 1, "exp": 0},
    },
    complexity_of_operators={"/": 2, "exp": 3},
    complexity_of_constants=2,
    select_k_features=4,
    progress=True,
    weight_randomize=0.1,
    cluster_manager=None,
    precision=64,
    warm_start=True,
    turbo=True,
    julia_project=None,
    update=False,
    extra_sympy_mappings={"cos2": lambda x: sympy.cos(x)**2},
)
'''
# ---------------------------------------------------Model 2-------------------------------------------------
'''

if __name__ == '__main__':
    # read table at first    
    df = parse_table("/data/hongcezh/clause-learning/data-collect/stat/size20.csv")
    # prepare data
    #X = 2 * np.random.randn(100, 5)
    
    # convert the dataframe to numpy array, X = [M, I, L, O, A]
    X = df.iloc[:, 1:6].to_numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # y = [n_clauses]
    y = df.iloc[:, -1].to_numpy()
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    model.fit(X_train, y_train)
    
    # Evaluate model performance on test data
    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred)**2)
    mae = np.mean(np.abs(y_test - y_pred))
    print(f"MSE: {mse}, MAE: {mae}")


    # print(model)
    
    # y_predict = model.predict(X)
    # print(y_predict)
    
    
    '''
    # validation
    model = PySRRegressor.from_file("hall_of_fame_2023-03-20_180019.134.pkl")
    print(model)
    # predict y by feeding X to the model
    y = model.predict(X)
    print(y)
    '''