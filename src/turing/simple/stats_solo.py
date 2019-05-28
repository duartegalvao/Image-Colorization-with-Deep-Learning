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

tr = np.mean(ds['correct'] == True)

false_positives = np.sum((ds['is_truth'] == False) & (ds['correct'] == False))
false_negatives = np.sum((ds['is_truth'] == True) & (ds['correct'] == False))

fpr = np.mean((ds['is_truth'] == False) & (ds['correct'] == False))
fnr = np.mean((ds['is_truth'] == True) & (ds['correct'] == False))
tpr = np.mean((ds['is_truth'] == True) & (ds['correct'] == True))
tnr = np.mean((ds['is_truth'] == False) & (ds['correct'] == True))

time_mean = np.mean(ds['time'])
time_std = np.std(ds['time'])

print("{} correct guesses, {} incorrect.".format(correct_n, incorrect_n))
print("{} false positives and {} false negatives.".format(false_positives, false_negatives))
print("FPR: {:.2f}%, FNR: {:.2f}%, TPR: {:.2f}%, TNR: {:.2f}%".format(fpr*100, fnr*100, tpr*100, tnr*100))
print("TR: {:.2f}%, FR: {:.2f}%".format(tr * 100, (1-tr) * 100))
print("Mean time: {:.0f} ± {:.0f} ms".format(time_mean, time_std))
