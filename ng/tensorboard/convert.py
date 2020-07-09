#!/usr/bin/env python3
import csv
import sys
import os
from tensorboard.backend.event_processing import event_accumulator


def tensorboard_to_csv(out_dir):
    event_acc = event_accumulator.EventAccumulator(out_dir)
    event_acc.Reload()

    metrics = event_acc.Tags()["scalars"]

    for m in metrics:
        dicts = list()
        for s in event_acc.Scalars(m):
            dicts.append({"step": s.step, "value": s.value})

        # This option is for appending multiple logs without the overhead
        # of processing the original log(s) again. Note that you'll need
        # to move the original log to a different directory first due
        # to Tensorboard's behavior.
        prev_exist = os.path.exists(f"{out_dir}/{m}.csv")

        output = open(f"{out_dir}/{m}.csv", "a+")
        writer = csv.DictWriter(output, fieldnames=list(dicts[0].keys()))

        if not prev_exist:
            writer.writeheader()
        for r in dicts:
            writer.writerow(r)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    tensorboard_to_csv(path)
