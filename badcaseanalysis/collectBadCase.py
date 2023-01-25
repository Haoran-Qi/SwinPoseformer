import json

rsnF = './data/rsn_test_compare.json'

f = open(rsnF)
data = json.load(f)

badCases = []

for d in data:
    if d['guess'] == 95 and d['label'] != 95:
        badCases.append(d)

with open("./data/badcase_95.json", "w") as outfile:
    json.dump(badCases, outfile, indent=4)