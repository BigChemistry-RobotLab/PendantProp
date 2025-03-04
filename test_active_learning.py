import pandas as pd
from analysis.active_learning import ActiveLearner
from analysis.models import szyszkowski_model
parameters = ["cmc", "gamma_max", "Kad"]
results = pd.read_csv("results.csv")
active_learner = ActiveLearner(model=szyszkowski_model, parameters=parameters)
c, st = active_learner.suggest(results=results, solution_name="SDS_2")
print(c)
print(type(c))