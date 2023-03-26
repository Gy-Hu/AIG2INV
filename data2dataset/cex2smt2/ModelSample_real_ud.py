# Real uniform distribution implementation

from z3 import *
from pycryptosat import Solver
import random

# Read the SMT2 file
with open('input.smt2', 'r') as f:
    input_smt2 = f.read()

# Parse the SMT2 content
s = Solver()
s.from_string(input_smt2)

# Get all variables
variables = [d for d in s.decls() if d.arity() == 0]

# Create a CryptoMiniSat instance
cms_solver = Solver()

# Add the clauses from Z3's solver
for clause in s.clauses():
    cms_solver.add_clause(clause)

# Function to generate a random model
def random_model(variables, cms_solver):
    assumptions = [(v, random.choice([True, False])) for v in variables]
    sat, solution = cms_solver.solve(assumptions)
    if sat:
        model = {}
        for v, value in zip(variables, solution):
            model[v] = value
        return model
    else:
        return None

# Generate a random model
model = random_model(variables, cms_solver)
if model is not None:
    print("Model found:")
    for var, value in model.items():
        print(f"{var} = {value}")
else:
    print("No model found.")
