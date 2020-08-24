import argparse

from retrain import utils
import analysis.benchmark as bench
from analysis import charts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default=None, help="prefix of model to test")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--in_list", default=None)
    parser.add_argument("--avg", action="store_true", default=False)
    parser.add_argument("--roll_avg", type=int, default=None)
    parser.add_argument("--tabulate", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--visualize_conf", default=None)
    parser.add_argument("--filter_sample", action="store_true", default=False)
    parser.add_argument("--compare_init", action="store_true", default=False)
    parser.add_argument("--metric", default="prec")
    parser.add_argument("--metric2", default=None)
    opt = parser.parse_args()

    config = utils.parse_retrain_config(opt.config)

    prefixes = [
        "init",
        "median-below-thresh",
        "median-thresh",
        "normal",
        "iqr",
        "mid-thresh",
        "mid-below-thresh",
        "mid-normal",
        "bin-quintile",
        "bin-normal",
        "random",
    ]

    if opt.benchmark and opt.prefix is None:
        for prefix in prefixes:
            bench.benchmark_batch_set(prefix, config, opt.roll_avg)
            bench.series_benchmark(config, prefix, avg=opt.avg, roll=opt.roll_avg)
    if opt.tabulate:
        if opt.prefix is not None:
            charts.tabulate_batch_samples(
                config, opt.prefix, filter=opt.filter_sample, roll=opt.roll_avg
            )
        else:
            charts.compare_benchmarks(
                config,
                prefixes,
                opt.metric,
                opt.metric2,
                roll=opt.roll_avg,
                compare_init=opt.compare_init,
                filter_sample=opt.filter_sample,
            )

    elif opt.visualize_conf:
        charts.visualize_conf(
            opt.prefix, opt.visualize_conf, opt.filter_sample, config["pos_thres"]
        )

    elif opt.prefix is not None:
        bench.benchmark_batch_set(opt.prefix, config, opt.roll_avg)
        charts.tabulate_batch_samples(config, opt.prefix, roll=opt.roll_avg)
        if opt.prefix != "init":
            bench.series_benchmark(config, opt.prefix, avg=opt.avg, roll=opt.roll_avg)
            charts.aggregate_results(
                config, opt.prefix, opt.metric, avg=opt.avg, roll=opt.roll_avg
            )
