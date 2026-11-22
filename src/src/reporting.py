import csv
import os
import statistics

from evaluate import HITS_K_VALUES

METRIC_NAMES = ["auc"] + ["hits@" + str(k) for k in HITS_K_VALUES]


def save_rows(rows, file_path):
    # Writes all result rows to a CSV file.
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", newline="") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=["split", "method"] + METRIC_NAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(rows, method_names):
    # Prints the mean and spread over random seeds, then each time split on its own.
    print()
    print("RANDOM SPLIT (mean +- std over seeds)")
    for method_name in method_names:
        parts = [method_name.ljust(20)]
        for metric in METRIC_NAMES:
            values = []
            for row in rows:
                if row["split"].startswith("random") and row["method"] == method_name:
                    values.append(row[metric])
            parts.append(metric + "=%.3f+-%.3f" % (statistics.mean(values), statistics.stdev(values)))
        print("  ".join(parts))

    time_split_names = []
    for row in rows:
        if row["split"].startswith("time") and row["split"] not in time_split_names:
            time_split_names.append(row["split"])
    for split_name in time_split_names:
        print()
        print(split_name.upper())
        for row in rows:
            if row["split"] != split_name:
                continue
            parts = [row["method"].ljust(20)]
            for metric in METRIC_NAMES:
                parts.append(metric + "=%.3f" % row[metric])
            print("  ".join(parts))