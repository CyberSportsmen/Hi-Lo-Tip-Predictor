import numpy as np
import matplotlib.pyplot as plt


day_map = {
    "Thur": 0,
    "Fri": 1,
    "Sat": 2,
    "Sun": 3
}

# mandria mea de functie
def is_weekend(day):
    return day == 2 or day == 3

def ReadData(path):
    data = []
    with open(path, "r") as file:
        next(file) # sarim peste prima linie, e gunoi
        for line in file:
            line = line.strip()
            if line:
                line = line.split(',')
                line[0] = float(line[0])
                line[1] = float(line[1])
                line[2] = line[2] == "Male" # male = 1, female = 0
                line[3] = line[3] == "Yes" # smoker = 1, non-smoker = 0
                line[4] = day_map[line[4]] # un fel de enum :)))
                line[5] = line[5] == "Dinner" # dinner = 1, lunch = 0
                line[6] = int(line[6])
                line.append(is_weekend(line[4]))
                data.append(line)
    return data

def SplitData(data, ratio=0.8):
    np.random.shuffle(data) # as vrea sa fie diferit de fiecare data antrenamentul
    index = int(len(data) * ratio)
    train = data[:index]
    test = data[index:]
    return train, test

# returneaza parametrii
def TrainParams(train_data):
    counts = {0: 0, 1: 0}
    feature_counts = {0: {}, 1: {}}
    numeric_sums = {0: {}, 1: {}}
    numeric_sq_sums = {0: {}, 1: {}}
    numeric_features = [0, 6]  # index pt total_bill, size

    for row in train_data:
        total, tip, sex, smoker, day, time, size, is_weekend = row
        label = (tip / total > 0.15)
        counts[label] += 1

        #numeric stats
        for i in numeric_features:
            val = row[i]
            numeric_sums[label][i] = numeric_sums[label].get(i, 0) + val
            numeric_sq_sums[label][i] = numeric_sq_sums[label].get(i, 0) + val ** 2

        #categorical counts
        features = [sex, smoker, day, time, is_weekend]
        for i, val in enumerate(features):
            j = i + 2  # sarim peste 0 si 1 pentru numeric
            if j not in feature_counts[label]:
                feature_counts[label][j] = {}
            feature_counts[label][j][val] = feature_counts[label][j].get(val, 0) + 1

    total_samples = np.sum(counts.values())
    params = {0: {}, 1: {}}

    for c in [0, 1]:
        params[c]["prior"] = counts[c] / total_samples
        params[c]["features"] = {}
        params[c]["gaussian"] = {}

        # pt valori continue
        for i in numeric_features:
            mean = numeric_sums[c][i] / counts[c]  # media
            var = (numeric_sq_sums[c][i] / counts[c]) - mean ** 2 # varianta
            params[c]["gaussian"][i] = (mean, var)

        # restul
        for i, val_counts in feature_counts[c].items():
            total_vals = sum(val_counts.values())
            unique_vals = len(val_counts)
            params[c]["features"][i] = {
                v: ((val_counts[v] + 1) / (total_vals + unique_vals)) for v in val_counts
            }

    return params
# formula de pe net, pare ca merge (astept cursu despre asta :PP)
def gaussian_prob(x, mean, var):
    if var == 0:
        return 1e-6  # sa nu impartim la zero :(
    exponent = np.exp(-((x - mean) ** 2) / (2 * var))
    return (1 / np.sqrt(2 * np.pi * var)) * exponent

def Predict(row, params):
    total, tip, sex, smoker, day, time, size, is_weekend = row
    features = [sex, smoker, day, time, is_weekend]
    numeric = [total, size]
    numeric_indices = [0, 6] #indicii valorilor numerice

    log_probs = {} # logaritmam pentru a nu avea probleme cu sanse foarte mici 
                   # (nu ar avea cum sa fie cazul aici, dar e best practice)

    for c in [0, 1]:
        log_prob = np.log(params[c]["prior"])

        # pt valori continue
        for i, x in zip(numeric_indices, numeric):
            mean, var = params[c]["gaussian"][i]
            prob = gaussian_prob(x, mean, var)
            log_prob += np.log(prob + 1e-9)

        # restul
        for i, val in enumerate(features):
            j = i + 2
            feature_probs = params[c]["features"].get(j, {})
            prob = feature_probs.get(val, 1e-6)
            log_prob += np.log(prob)

        log_probs[c] = log_prob

    return log_probs[1] > log_probs[0]

def Evaluate(test_data, params):
    corect = 0
    for row in test_data:
        total, tip, _ = row
        actual = (tip / total > 0.15)
        predicted = Predict(row, params)
        if actual == predicted:
            corect += 1
    accuracy = corect / len(test_data)
    return accuracy


if __name__ == "__main__":
    path = "./tip.csv"
    data = ReadData(path)
    train, test = SplitData(data)

    print(f"Training set size: {len(train)}")
    print(f"Testing set size: {len(test)}")
    nr_runs = 1000
    accuracies = [Evaluate(SplitData(data)[1], TrainParams(SplitData(data)[0])) for _ in range(nr_runs)]
    print(f"Average accuracy over {nr_runs} runs: {np.mean(accuracies):.2%}")

