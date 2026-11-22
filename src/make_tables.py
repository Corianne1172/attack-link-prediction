import csv
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RESULT_FILES = [os.path.join("results", "baselines.csv"), os.path.join("results", "embeddings.csv")]
SUMMARY_FILE = os.path.join("results", "summary.csv")
LATEX_FILE = os.path.join("results", "results_table.tex")
FIGURE_DIR = os.path.join("results", "figures")

METHOD_ORDER = ["popularity", "common_neighbors", "jaccard", "adamic_adar", "three_hop_paths",
                "similar_groups", "deepwalk_cosine", "deepwalk_context", "deepwalk_classifier"]
METHOD_NAMES = {
    "popularity": "Popularity",
    "common_neighbors": "Common Neighbors",
    "jaccard": "Jaccard",
    "adamic_adar": "Adamic-Adar",
    "three_hop_paths": "3-hop paths",
    "similar_groups": "Similar groups",
    "deepwalk_cosine": "DeepWalk (cosine)",
    "deepwalk_context": "DeepWalk (context)",
    "deepwalk_classifier": "DeepWalk (classifier)",
}
GROUPS = ["random", "time_2023", "time_2024"]
GROUP_LABELS = {"random": "Random split", "time_2023": "Time split (2023)", "time_2024": "Time split (2024)"}
METRICS = ["auc", "hits@10", "hits@50"]
METRIC_LABELS = {"auc": "AUC", "hits@10": "H@10", "hits@50": "H@50"}

# Colors: the first three slots of a validated categorical palette, plus chart ink and surface colors
GROUP_COLORS = {"random": "#2a78d6", "time_2023": "#eb6834", "time_2024": "#1baf7a"}
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def load_rows(file_paths):
    # Reads the result CSV files into one list of rows.
    rows = []
    for file_path in file_paths:
        with open(file_path, "r") as result_file:
            reader = csv.DictReader(result_file)
            for row in reader:
                rows.append(row)
    return rows


def get_group_name(split_name):
    # Maps a split name like random_seed3 to its group, random. Time splits keep their own name.
    if split_name.startswith("random"):
        return "random"
    return split_name


def summarize(rows):
    # Returns (mean, spread) for each (method, group, metric). Spread is None when there is one run.
    values = {}
    for row in rows:
        group = get_group_name(row["split"])
        for metric in METRICS:
            key = (row["method"], group, metric)
            values.setdefault(key, []).append(float(row[metric]))
    summary = {}
    for key, number_list in values.items():
        mean = statistics.mean(number_list)
        if len(number_list) > 1:
            spread = statistics.stdev(number_list)
        else:
            spread = None
        summary[key] = (mean, spread)
    return summary


def format_value(mean, spread):
    # Writes a value as text, with the spread when there is one.
    if spread is None:
        return "%.3f" % mean
    return "%.3f+-%.3f" % (mean, spread)


def save_summary(summary, file_path):
    # Writes the summary to a CSV file with one row per method, group and metric.
    with open(file_path, "w", newline="") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(["method", "group", "metric", "mean", "std"])
        for method in METHOD_ORDER:
            for group in GROUPS:
                for metric in METRICS:
                    mean, spread = summary[(method, group, metric)]
                    if spread is None:
                        spread_text = ""
                    else:
                        spread_text = "%.4f" % spread
                    writer.writerow([method, group, metric, "%.4f" % mean, spread_text])


def print_tables(summary):
    # Prints one table per group of splits.
    for group in GROUPS:
        print()
        print(GROUP_LABELS[group])
        for method in METHOD_ORDER:
            parts = [METHOD_NAMES[method].ljust(22)]
            for metric in METRICS:
                mean, spread = summary[(method, group, metric)]
                parts.append(METRIC_LABELS[metric] + "=" + format_value(mean, spread))
            print("  ".join(parts))


def write_latex_table(summary, file_path):
    # Writes the results as a LaTeX table with the best value in each column in bold.
    columns = []
    for group in GROUPS:
        for metric in METRICS:
            columns.append((group, metric))

    best_values = {}
    for column in columns:
        group, metric = column
        best_value = 0.0
        for method in METHOD_ORDER:
            mean = summary[(method, group, metric)][0]
            if mean > best_value:
                best_value = mean
        best_values[column] = best_value

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Link prediction results. Random-split values are means over 5 seeds, "
                 r"time-split values are single runs. H@$k$ is hits@$k$. The best value in each column is in bold.}")
    lines.append(r"\label{tab:results}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lccccccccc}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{3}{c}{Random split} & \multicolumn{3}{c}{Time split (2023)} "
                 r"& \multicolumn{3}{c}{Time split (2024)} \\")
    lines.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}")
    lines.append(r"Method & AUC & H@10 & H@50 & AUC & H@10 & H@50 & AUC & H@10 & H@50 \\")
    lines.append(r"\midrule")
    for method in METHOD_ORDER:
        # A thin rule separates the heuristics from the DeepWalk variants
        if method == "deepwalk_cosine":
            lines.append(r"\midrule")
        cells = [METHOD_NAMES[method]]
        for column in columns:
            group, metric = column
            mean = summary[(method, group, metric)][0]
            cell = "%.3f" % mean
            if mean == best_values[column]:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    with open(file_path, "w") as latex_file:
        latex_file.write("\n".join(lines) + "\n")


def make_figure(summary, png_path, pdf_path):
    # Draws grouped horizontal bars: AUC on the left, hits@10 on the right.
    plt.rcParams["font.family"] = "sans-serif"
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    figure.patch.set_facecolor(SURFACE)

    panels = [("auc", "AUC", (0.0, 1.0)), ("hits@10", "Hits@10", (0.0, 0.35))]
    bar_height = 0.22
    group_offsets = {"random": -0.25, "time_2023": 0.0, "time_2024": 0.25}

    for panel_index in range(len(panels)):
        axis = axes[panel_index]
        metric, title, x_limits = panels[panel_index]
        axis.set_facecolor(SURFACE)
        for method_index in range(len(METHOD_ORDER)):
            method = METHOD_ORDER[method_index]
            for group in GROUPS:
                mean, spread = summary[(method, group, metric)]
                y_position = method_index + group_offsets[group]
                axis.barh(y_position, mean, height=bar_height, color=GROUP_COLORS[group], zorder=3)
                # Error bars only exist for the random split, which has 5 seeds
                if spread is not None:
                    axis.errorbar(mean, y_position, xerr=spread, fmt="none", ecolor=INK_SECONDARY,
                                  elinewidth=1, capsize=2, zorder=4)

        axis.set_xlim(x_limits)
        axis.set_title(title, loc="left", color=INK_PRIMARY, fontsize=12)
        axis.xaxis.grid(True, color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ["top", "right", "left"]:
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(BASELINE)
        axis.tick_params(axis="both", colors=INK_SECONDARY, length=0)
        # A faint line separates the heuristics (above) from the DeepWalk variants (below)
        axis.axhline(5.5, color=GRID, linewidth=0.8)

    # A random guess scores 0.5 AUC, so mark it on the AUC panel
    axes[0].axvline(0.5, color=BASELINE, linewidth=1, zorder=1)
    axes[0].text(0.51, -0.65, "chance", color=INK_SECONDARY, fontsize=9, va="center")

    axes[0].set_yticks(range(len(METHOD_ORDER)))
    method_labels = []
    for method in METHOD_ORDER:
        method_labels.append(METHOD_NAMES[method])
    axes[0].set_yticklabels(method_labels)
    axes[0].set_ylim(len(METHOD_ORDER) - 0.5, -1.0)

    legend_handles = []
    for group in GROUPS:
        legend_handles.append(Patch(color=GROUP_COLORS[group], label=GROUP_LABELS[group]))
    figure.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, labelcolor=INK_SECONDARY)
    figure.tight_layout(rect=(0, 0.06, 1, 1))

    figure.savefig(png_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    figure.savefig(pdf_path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)


def main():
    rows = load_rows(RESULT_FILES)
    summary = summarize(rows)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    save_summary(summary, SUMMARY_FILE)
    write_latex_table(summary, LATEX_FILE)
    make_figure(summary, os.path.join(FIGURE_DIR, "results.png"), os.path.join(FIGURE_DIR, "results.pdf"))
    print_tables(summary)
    print()
    print("Saved summary.csv, results_table.tex and figures/results.png in results/")


if __name__ == "__main__":
    main()