import argparse
import glob

from retrain.dataloader import ImageFolder
import retrain.utils as utils
import retrain.benchmark as bench


def series_benchmark(config, prefix):
    # TODO: Write this
    # 1. Find the number of batches for the given prefix
    # 2. Find the starting/ending epochs of each split
    # 3. Benchmark that itertion's test set with the average method
    #    (Could plot this, but may not be meaningful due to differing test sets)
    # 4. Benchmark the overall test set with the same average method (and save results)
    #    4a. plot the overall test set performance as a function of epoch number
    # 5. (optional) serialize results of the overall test set as JSON for improved speed when using averages
    init_test_set = f"{config['output']}/init_test.txt"
    combined_test = ImageFolder(init_test_set, config["img_size"])
    test_sets = sorted(glob.glob(f"{config['output']}/{prefix}*_test.txt"))

    benchmark_avg(img_folder, prefix, start, end, total, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start", required=True, type=int, help="starting benchmark epoch",
    )
    parser.add_argument(
        "--end", required=True, type=int, help="ending benchmark epoch",
    )
    parser.add_argument(
        "--delta", type=int, help="interval to plot", default=3, required=False
    )
    parser.add_argument("--prefix", default="init", help="prefix of model to test")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--in_list", default=None)
    opt = parser.parse_args()

    config = utils.parse_retrain_config(opt.config)
    images = utils.get_lines(opt.in_list)
    img_folder = ImageFolder(images, config["img_size"], prefix=opt.prefix)

    bench.series_benchmark_loss(
        img_folder, opt.prefix, opt.start, opt.end, opt.delta, config, opt.out
    )

    # for test in ("", "test_"):
    #     output = open(f"{opt.output}/val_precision_{test}time.csv", "w+")
    #     output.write("epoch,all_precision\n")
    #     for i in tqdm(
    #         range(opt.start, opt.end, opt.delta), f"Benchmarking {test} results"
    #     ):
    #         if not os.path.exists(f"{opt.output}/benchmark_{test}{i}.csv"):
    #             benchmark.benchmark(
    #                 opt.prefix,
    #                 i,
    #                 "config/yolov3.cfg",
    #                 "config/chars.data",
    #                 "config/chars.names",
    #                 "data/images/objs/" if test == str() else "data/temp/",
    #                 out=f"{opt.output}/benchmark_{test}",
    #                 silent=True,
    #             )

    #         results, _ = utils.load_data(
    #             f"{opt.output}/benchmark_{test}{i}.csv", by_actual=True
    #         )
    #         output.write(f"{i},{benchmark.mean_precision(results[:-1])}\n")

    #     output.close()
