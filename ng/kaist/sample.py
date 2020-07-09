import random
import benchmark
import sys
import statistics


OUTPUT = "./output/"


def sample(result, desired, prob_func, *prob_func_args):
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
            conf = float(row["conf"])
            choose = random.random() <= prob_func(conf, *prob_func_args)
            if choose:
                chosen_this_round.append(row)
        chosen += chosen_this_round
        for row in chosen_this_round:
            pool.remove(row)

    chosen = chosen[:desired]

    hit_rate = sum([int(row["hit"] == "True") for row in chosen]) / desired
    print(f"Hit rate for {result.name}: {hit_rate}")

    return chosen


def const(conf, thresh):
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
    return len([res for res in result.get_all() if float(res["conf"]) >= thresh])


if __name__ == "__main__":
    random.seed("sage")
    results, _ = benchmark.load_data(sys.argv[1], by_actual=False)
    retrain_by_class = list()

    for result in results:
        confidences = result.get_confidences()
        avg = sum(confidences) / len(confidences)
        median = statistics.median(confidences)
        print(f"avg: {avg}")
        print(f"median: {median}")

        retrain_by_class.append(
            sample(result, above_thresh(result, median), const, median)
        )

    max_samp = float("inf")
    for class_list in retrain_by_class:
        max_samp = min(max_samp, len(class_list))

    print(f"Samples per class: {max_samp}")

    retrain = list()
    for sample_list in retrain_by_class:
        retrain += sample_list[:max_samp]

    with open(OUTPUT + "retrain.txt", "w+") as out:
        files = [result["file"] for result in retrain]
        out.write("\n".join(files) + "\n")

    create_labels(retrain)
