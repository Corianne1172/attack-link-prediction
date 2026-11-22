import csv
import os
import statistics

from splits import NODE_FILE, EDGE_FILE, load_nodes, load_edges, build_all_splits
from heuristics import build_scoring_context, SCORE_FUNCTIONS
from evaluate import compute_auc, compute_hits, HITS_K_VALUES

RESULT_FILE = os.path.join("results", "baselines.csv")
METRIC_NAMES = ["auc"] + ["hits@" + str(k) for k in HITS_K_VALUES]


def evaluate_split(split):
    # Runs every baseline on one split and returns one result row per method.
    context = build_scoring_context(split)
    rows = []
    for method_name, score_function in SCORE_FUNCTIONS.items():
        hits = compute_hits(score_function, context, split, 0)
        row = {
            "split": split["name"],
            "method": method_name,
            "auc": compute_auc(score_function, context, split),
        }
        for k in HITS_K_VALUES:
            row["hits@" + str(k)] = hits[k]
        rows.append(row)
        print(split["name"], method_name, "done")
    return rows


def save_rows(rows, file_path):
    # Writes all result rows to a CSV file.
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", newline="") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=["split", "method"] + METRIC_NAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(rows):
    # Prints the mean and spread over random seeds, then each time split on its own.
    print()
    print("RANDOM SPLIT (mean +- std over seeds)")
    for method_name in SCORE_FUNCTIONS:
        parts = [method_name.ljust(18)]
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
            parts = [row["method"].ljust(18)]
            for metric in METRIC_NAMES:
                parts.append(metric + "=%.3f" % row[metric])
            print("  ".join(parts))


def main():
    nodes = load_nodes(NODE_FILE)
    edges = load_edges(EDGE_FILE)
    all_rows = []
    for split in build_all_splits(nodes, edges):
        all_rows.extend(evaluate_split(split))
    save_rows(all_rows, RESULT_FILE)
    print_summary(all_rows)


if __name__ == "__main__":
    main()