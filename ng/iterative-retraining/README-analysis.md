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
python3 analyze.py --config <retrain config file> [--benchmark [--avg | --roll <epoch span> --delta <epoch span>]] \
	(--prefix <sampling method> | --tabulate | --visualize <benchmark> | --view_benchmark <benchmark>) \
	[--filter_sample --compare_init --batch_test <test set size> --aggr_median] [--metric <metric name> --metric2 <metric name>]
```

The only required argument is `--config` along with the path to the pipeline configuration file. However, for the script to produce a visual output, one of the additional flags in parentheses must be specified. The `--benchmark` flag may also be set to generate benchmark files if they have not yet been generated.

Many flags also have additional optional parameters for filtering out data.

## Benchmarking

There are two main types of benchmarking, both of which are conducted when the `--benchmark` flag is specified:

* **Series benchmarking**
* **Batch split benchmarking**

Aside from filename differences, all benchmark files have contents that follow the [benchmarks generated from the sampling pipeline](./README.md#training-output).

Additional flags that affect benchmarking include:

* `--prefix`: if this is specified, benchmarking will only occur on the specified sampling method.
* `--avg`:
* `--roll_avg <epoch span>`:
* `--delta`:
* `--batch_test <test set size>`:


## Training Curves

## Sampling Method Metric Tables

### Comparing Metrics

## Benchmark Metric Table

## Benchmark Confidence Histogram