#!/usr/bin/env python3
import csv
from tensorboard.backend.event_processing import event_accumulator


def tensorboard_to_csv(out_dir):
    event_acc = event_accumulator.EventAccumulator(out_dir)
    event_acc.Reload()

    metrics = event_acc.Tags()["scalars"]

    for m in metrics:
        dicts = list()
        for s in event_acc.Scalars(m):
            dicts.append({"step": s.step, "value": s.value})

        output = open(f"{out_dir}/{m}.csv", "a+")
        writer = csv.DictWriter(output, fieldnames=list(dicts[0].keys()))
        writer.writeheader()
        for r in dicts:
            writer.writerow(r)


if __name__ == "__main__":
    tensorboard_to_csv(".")
