import sys
import benchmark
import csv
from sklearn.metrics import confusion_matrix


def load_data(output, by_actual=True):
    samples = dict()
    all_data = list()

    actual = list()
    pred = list()

    with open(output, newline="\n") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            actual.append(row["actual"])
            pred.append(row["detected"])

            key_val = row["actual"] if by_actual else row["detected"]
            if key_val == str():
                continue
            if key_val not in samples.keys():
                samples[key_val] = [row]
            else:
                samples[key_val].append(row)
            all_data.append(row)
    samples = {k: samples[k] for k in sorted(samples)}
    results = [benchmark.ClassResults(k, v) for k, v in samples.items()]
    mat = confusion_matrix(actual, pred, labels=list(samples.keys()) + [""])

    results.append(benchmark.ClassResults("All", all_data))

    return results, mat


def save_stdout(filename, func, *pos_args, **var_args):
    old_stdout = sys.stdout
    sys.stdout = open(filename, "w+")
    func(*pos_args, **var_args)
    sys.stdout = old_stdout


def rewrite_test_list(list_path, orig_data, name="test-new.txt"):
    with open(list_path, "r") as valid:
        new_valid = valid.read().replace("data/", orig_data)
        with open(list_path.replace("test.txt", name), "w+") as new_valid_file:
            new_valid_file.write(new_valid)
