import numpy as np
import json
import mlflow
import sys


model = 'mlruns/1/70439eb482b54d56b54b0ecc6f1ca96f/artifacts/s444409'
model = mlflow.pyfunc.load_model(model)

example = sys.argv[1]
data_p = np.array([example['inputs'][0]], dtype=np.float32)
print(10*'=' + 'PREDICTIONS' + 10*'=')
print({model.predict(data_p)})