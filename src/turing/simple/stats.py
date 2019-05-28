import json
import sys

import pandas as pd
import numpy as np


filename = "tests.json"

if len(sys.argv) > 1:
    filename = sys.argv[1]

with open(filename) as json_file:
    data = json.load(json_file)

ds = pd.read_json(json.dumps(data['tests']), 'records')

correct_n = np.sum(ds['correct'] == True)
incorrect_n = np.sum(ds['correct'] == False)
success_rate = np.mean(ds['correct'] == True)

time_mean = np.mean(ds['time'])
time_std = np.std(ds['time'])

print("{} correct guesses, {} incorrect.".format(correct_n, incorrect_n))
print("Success rate: {}%".format(success_rate * 100))
print("Mean time: {} ± {} ms".format(time_mean, time_std))
