import random
import benchmark
import sys


OUTPUT = "./output/"


def sample(result, desired, prob_func):
    pool = result.get_all()
    random.shuffle(pool)
    chosen = list()

    while len(chosen) < desired:
        chosen_this_round = list()
        for row in pool:
            choose = random.random() <= prob_func(float(row["conf"]))
            if choose:
                chosen_this_round.append(row)
        chosen += chosen_this_round
        for row in chosen_this_round:
            pool.remove(row)

    files = [row["file"] for row in chosen[:desired]]
    return files


def const(conf):
    if conf >= 0.5:
        return 1.0
    return 0.0


if __name__ == "__main__":
    # NOTE: Load data currently loads data by *actual* class,
    # not *predicted* class. This should be changed.
    random.seed("sage")
    results, _ = benchmark.load_data(sys.argv[1])
    retrain = list()

    for result in results:
        retrain += sample(result, int(0.75 * len(result.get_all())), const)

    with open(OUTPUT + "retrain.txt", "w+") as out:
        out.write("\n".join(retrain))
