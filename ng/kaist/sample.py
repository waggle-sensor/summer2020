import random
import benchmark
import sys
import statistics
import os
import utils
import yolov3.utils.parse_config as parser


OUTPUT = "./output/"
ORIG_DATA = "../yolov3/data/"


def prob_sample(result, desired, prob_func, *prob_func_args):
    """Generate a list of files for sampling.
    result:     a ClassResult holding a list of images
    desired:    number of samples to extract per class
    prob_func:  a function that takes a confidence score as an input and
                outputs the probability of sampling the image with that confidence

    The function continues sampling until the desired number of samples is hit.
    Consequently, the probability function should be well-chosen to prevent long runtimes.
    """
    pool = result.get_all()
    random.shuffle(pool)
    chosen = list()

    while len(chosen) < desired:
        chosen_this_round = list()
        for row in pool:
            conf = row["conf"]
            choose = random.random() <= prob_func(conf, *prob_func_args)
            if choose:
                chosen_this_round.append(row)
        chosen += chosen_this_round
        for row in chosen_this_round:
            pool.remove(row)

    chosen = chosen[:desired]

    # hit_rate = sum([int(row["hit"] == "True") for row in chosen]) / desired
    # print(f"Hit rate for {result.name}: {hit_rate}")

    return chosen


def median_thresh_sample(result):
    confidences = result.get_confidences()
    avg = sum(confidences) / len(confidences)
    median = statistics.median(confidences)

    print(f"avg: {avg}")
    print(f"median: {median}")

    return prob_sample(result, above_thresh(result, median), const, median)


def const(conf, thresh=0.5):
    if conf >= thresh:
        return 1.0
    return 0.0


def create_labels(retrain_list):
    classes = open("config/chars.names").read().split("\n")[:-1]
    for result in retrain_list:
        idx = classes.index(result["detected"])
        with open(result["file"].replace(".png", ".txt"), "w+") as label:
            label.write(f"{idx} 0.5 0.5 1 1")


def above_thresh(result, thresh):
    """Get the number of elements in a ClassResult above a threshold."""
    return len([res for res in result.get_all() if res["conf"] >= thresh])


def create_config(samples, sample_name, data_config):
    data_opts = parser.parse_data_config(data_config)

    new_valid = data_opts["valid"].replace(".txt", "-new.txt")
    if not os.path.exists(new_valid):
        utils.rewrite_test_list(data_opts["valid"], ORIG_DATA)

    config_path = OUTPUT + f"configs-retrain/{sample_name}/"
    os.makedirs(config_path, exist_ok=True)

    with open(config_path + "train.txt", "w+") as out:
        files = [result["file"] for result in samples]
        out.write("\n".join(files) + "\n")

    data_opts["train"] = config_path + "train-aug.txt"
    data_opts["valid"] = new_valid
    with open(config_path + data_config.split("/")[-1], "w+") as new_data_config:
        for k, v in data_opts.items():
            new_data_config.write(f"{k} = {v}\n")


def create_sample(data_file, results, sample_name, sample_func, *sample_func_args):
    retrain_by_class = list()

    print(f"===== {sample_name} ======")
    for result in results:
        if result.name == "All":
            continue
        retrain_by_class.append(sample_func(result, *sample_func_args))

    max_samp = float("inf")
    for class_list in retrain_by_class:
        max_samp = min(max_samp, len(class_list))

    print(f"Samples per class: {max_samp}")

    retrain = list()
    for sample_list in retrain_by_class:
        retrain += sample_list[:max_samp]

    hit_rate = sum(int(img["hit"] == "True") for img in retrain) / len(retrain)
    print(f"Hit rate: {hit_rate}")

    create_labels(retrain)
    create_config(retrain, sample_name, data_file)


if __name__ == "__main__":
    random.seed("sage")
    results, _ = utils.load_data(sys.argv[1], by_actual=False)
    create_sample("config/chars.data", results, "median-thresh", median_thresh_sample)
