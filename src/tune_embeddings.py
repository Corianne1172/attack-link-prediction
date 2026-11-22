import csv
import os

from embeddings import build_embedding_context, SCORE_FUNCTIONS, EMBEDDING_SETTINGS
from evaluate import compute_auc
from splits import NODE_FILE, EDGE_FILE, load_nodes, load_edges, make_random_split, make_validation_split

RESULT_FILE = os.path.join("results", "tuning.csv")
DIMENSIONS = [64]
WINDOWS = [40]
WALK_LENGTHS = [40, 80]
USE_SOFTWARE_OPTIONS = [True, False]
VALIDATION_SPLIT_SEEDS = [0, 1]


def make_validation_splits(nodes, edges):
    # Builds validation splits from the training part of the first two random splits.
    validation_splits = []
    for seed in VALIDATION_SPLIT_SEEDS:
        random_split = make_random_split(nodes, edges, seed)
        validation_splits.append(make_validation_split(random_split, seed))
    return validation_splits


def evaluate_setting(settings, validation_splits):
    # Returns each scoring method's validation AUC, averaged over the validation splits.
    auc_totals = {}
    for method_name in SCORE_FUNCTIONS:
        auc_totals[method_name] = 0.0
    for validation_split in validation_splits:
        context = build_embedding_context(validation_split, 0, settings)
        for method_name, score_function in SCORE_FUNCTIONS.items():
            auc_totals[method_name] += compute_auc(score_function, context, validation_split)
    mean_aucs = {}
    for method_name in SCORE_FUNCTIONS:
        mean_aucs[method_name] = auc_totals[method_name] / len(validation_splits)
    return mean_aucs


def get_auc(row):
    # Returns a result row's AUC, used for sorting.
    return row["auc"]


def describe_row(row):
    # Writes a result row's settings and AUC as one line of text.
    return ("dimension=%d window=%d walk_length=%d use_software=%s auc=%.3f" %
            (row["dimension"], row["window"], row["walk_length"], row["use_software"], row["auc"]))


def save_rows(rows, file_path):
    # Writes the tuning results to a CSV file.
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    field_names = ["dimension", "window", "walk_length", "use_software", "method", "auc"]
    with open(file_path, "w", newline="") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_best(rows):
    # Prints every result ranked, then the best setting for each scoring method.
    ranked = sorted(rows, key=get_auc, reverse=True)
    print()
    print("ALL RESULTS (validation AUC)")
    for row in ranked:
        print(row["method"].ljust(20), describe_row(row))
    print()
    print("BEST PER METHOD")
    for method_name in SCORE_FUNCTIONS:
        for row in ranked:
            if row["method"] == method_name:
                print(method_name.ljust(20), describe_row(row))
                break


def main():
    nodes = load_nodes(NODE_FILE)
    edges = load_edges(EDGE_FILE)
    validation_splits = make_validation_splits(nodes, edges)

    rows = []
    for dimension in DIMENSIONS:
        for window in WINDOWS:
            for walk_length in WALK_LENGTHS:
                for use_software in USE_SOFTWARE_OPTIONS:
                    settings = dict(EMBEDDING_SETTINGS)
                    settings["dimension"] = dimension
                    settings["window"] = window
                    settings["walk_length"] = walk_length
                    settings["use_software"] = use_software
                    mean_aucs = evaluate_setting(settings, validation_splits)
                    for method_name, auc in mean_aucs.items():
                        rows.append({"dimension": dimension, "window": window,
                                     "walk_length": walk_length, "use_software": use_software,
                                     "method": method_name, "auc": auc})
                    print("done: window", window, "use_software", use_software)

    save_rows(rows, RESULT_FILE)
    print_best(rows)


if __name__ == "__main__":
    main()