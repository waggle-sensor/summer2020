# Analysis Tool Usage

Aside from the [main module](./__main__.py) for starting the sampling and retraining pipeline, a [script for analyzing training results](./analyze.py) is included in this folder. The program supports the following features:

* Benchmarking against various test sets as a function of one or multiple averaged models and plotting the resultant training curves
* Tabulating performance metrics (precision, accuracy, recall, avg. confidence) for a particular benchmark
* Comparing metrics across various sampling methods using next-batch test methods
* Linear regression between two metrics to analyze correlation between sample features and performance improvements
* Generating confidence distributions of inference results for each sample batch
* Filtering sample batches for the specific images collected in that sample

To use the script, execute the following:

```
python3 analyze.py --config <retrain config file> [--benchmark [--avg --roll <epoch span> --delta <epoch span>]] \
	(--prefix <sampling method> | --tabulate | --visualize <benchmark> | --view_benchmark <benchmark>) \
	[--filter_sample --compare_init --batch_test <test set size>] [--metric <metric name> --metric2 <metric name>]
```

The only required argument is 