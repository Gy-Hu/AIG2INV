from z3 import *
import random
import argparse

def load_smt2_file(filename):
    with open(filename, "r") as file:
        content = file.read()
    return parse_smt2_string(content)

def get_random_seed():
    return random.randint(0, 2**32 - 1)

def uniform_random_sat_model(formula, num_models=1):
    models = []
    while len(models) < num_models:
        s = Solver()
        s.set("random_seed", get_random_seed())
        s.add(formula)
        if s.check() == sat:
            model = s.model()
            if model not in models:
                models.append(model)
    return models

if __name__ == "__main__":
    # add parser
    parser = argparse.ArgumentParser(description="Randomly generate uniform distribution of models")
    #parse smt2 file
    parser.add_argument('--smt2-file', type=str, default=None, help='The path to the smt2 file')
    parser.add_argument('--num-models', type=int, default=1, help='The number of models to generate')
    args = parser.parse_args()
    # Load the SMT2 file
    filename = args.smt2_file
    formula = load_smt2_file(filename)

    # Get the uniform random SAT models
    num_models = args.num_models
    models = uniform_random_sat_model(formula, num_models)

    # Print the models
    for idx, model in enumerate(models):
        print(f"Model {idx + 1}:")
        for d in model.decls():
            print(f"  {d.name()} = {model[d]}") #uniform distribution of models -> ludy's 