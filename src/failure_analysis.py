import csv
import os
import random

from cutoff_diagnostics import explain_cold_edges
from embeddings import build_embedding_context, deepwalk_cosine_score
from evaluate import group_held_out_techniques
from heuristics import popularity_score, three_hop_paths_score, similar_groups_score
from splits import NODE_FILE, EDGE_FILE, load_nodes, load_edges, build_all_splits

RESULT_FILE = os.path.join("results", "failure_analysis.csv")
SPLIT_NAMES = ["random_seed0", "time_2023", "time_2024"]
RANK_CUTOFF = 50
NOT_RANKED = 10 ** 9

METHODS = {
    "popularity": popularity_score,
    "three_hop_paths": three_hop_paths_score,
    "similar_groups": similar_groups_score,
    "deepwalk_cosine": deepwalk_cosine_score,
}
TECHNIQUE_BUCKETS = ["unseen (0 groups)", "rare (1-2)", "medium (3-9)", "common (10+)"]
GROUP_BUCKETS = ["small (<10)", "medium (10-29)", "large (30+)"]


def compute_ranks(score_function, context, split, seed):
    # Returns the rank (1 is best) of every held-out technique among its group's candidates.
    rng = random.Random(seed)
    held_out_by_group = group_held_out_techniques(split["test_pairs"])
    ranks = {}
    for group_id, held_out in held_out_by_group.items():
        already_known = context["techniques_by_group"].get(group_id, set())
        ranked = []
        for technique_id in split["technique_ids"]:
            if technique_id in already_known:
                continue
            score = score_function(context, group_id, technique_id)
            # Random tie-breaker, so equal scores are not decided by the order of the ids
            ranked.append((-score, rng.random(), technique_id))
        ranked.sort()
        for position in range(len(ranked)):
            technique_id = ranked[position][2]
            if technique_id in held_out:
                ranks[(group_id, technique_id)] = position + 1
    return ranks


def get_technique_bucket(group_count):
    # Names the popularity bucket of a technique, from how many training groups use it.
    if group_count == 0:
        return TECHNIQUE_BUCKETS[0]
    if group_count <= 2:
        return TECHNIQUE_BUCKETS[1]
    if group_count <= 9:
        return TECHNIQUE_BUCKETS[2]
    return TECHNIQUE_BUCKETS[3]


def get_group_bucket(technique_count):
    # Names the size bucket of a group, from how many techniques it has in training.
    if technique_count < 10:
        return GROUP_BUCKETS[0]
    if technique_count < 30:
        return GROUP_BUCKETS[1]
    return GROUP_BUCKETS[2]


def recall_at_cutoff(edges, ranks):
    # Share of the edges whose technique was ranked within the cutoff. Returns None for no edges.
    if len(edges) == 0:
        return None
    hit_count = 0
    for edge in edges:
        if ranks.get(edge, NOT_RANKED) <= RANK_CUTOFF:
            hit_count += 1
    return hit_count / len(edges)


def analyze_buckets(split, analysis_name, bucket_by_edge, bucket_names, ranks_by_method):
    # Computes recall per bucket and method, prints the table, and returns the result rows.
    edges_by_bucket = {}
    for bucket_name in bucket_names:
        edges_by_bucket[bucket_name] = []
    for edge in split["test_pairs"]:
        edges_by_bucket[bucket_by_edge[edge]].append(edge)
    edges_by_bucket["all edges"] = list(split["test_pairs"])

    print()
    print(split["name"], "-", analysis_name, "(recall@%d)" % RANK_CUTOFF)
    print("   " + "bucket".ljust(20) + "edges".ljust(8) + "  ".join(name.ljust(16) for name in METHODS))
    rows = []
    for bucket_name in bucket_names + ["all edges"]:
        edges = edges_by_bucket[bucket_name]
        parts = []
        for method_name in METHODS:
            recall = recall_at_cutoff(edges, ranks_by_method[method_name])
            if recall is None:
                parts.append("-".ljust(16))
                recall_text = ""
            else:
                parts.append(("%.3f" % recall).ljust(16))
                recall_text = "%.4f" % recall
            rows.append([split["name"], analysis_name, bucket_name, len(edges), method_name, recall_text])
        print("   " + bucket_name.ljust(20) + str(len(edges)).ljust(8) + "  ".join(parts))
    return rows


def print_cold_start(split):
    # Prints how many future edges involve groups or techniques the training graph never saw.
    cold_count = len(split["cold_pairs"])
    if cold_count == 0:
        return
    usable_count = len(split["test_pairs"])
    total_count = usable_count + cold_count
    new_group, new_technique, both = explain_cold_edges(split)
    print()
    print(split["name"], "- cold-start")
    print("   later edges:", total_count, "| usable test edges:", usable_count,
          "| cold-start edges:", cold_count, "(%.0f%%)" % (100.0 * cold_count / total_count))
    print("   cold because of: new group", new_group, "| new technique", new_technique, "| both", both)


def save_rows(rows, file_path):
    # Writes all failure-analysis rows to a CSV file.
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", newline="") as result_file:
        writer = csv.writer(result_file)
        writer.writerow(["split", "analysis", "bucket", "edges", "method", "recall@50"])
        for row in rows:
            writer.writerow(row)


def main():
    nodes = load_nodes(NODE_FILE)
    edges = load_edges(EDGE_FILE)
    all_rows = []
    for split in build_all_splits(nodes, edges):
        if split["name"] not in SPLIT_NAMES:
            continue
        # The embedding context also holds everything the heuristics need
        context = build_embedding_context(split, 0)
        ranks_by_method = {}
        for method_name, score_function in METHODS.items():
            ranks_by_method[method_name] = compute_ranks(score_function, context, split, 0)

        technique_bucket_by_edge = {}
        group_bucket_by_edge = {}
        for group_id, technique_id in split["test_pairs"]:
            group_count = len(context["groups_by_technique"].get(technique_id, set()))
            technique_count = len(context["techniques_by_group"].get(group_id, set()))
            technique_bucket_by_edge[(group_id, technique_id)] = get_technique_bucket(group_count)
            group_bucket_by_edge[(group_id, technique_id)] = get_group_bucket(technique_count)

        all_rows.extend(analyze_buckets(split, "technique popularity", technique_bucket_by_edge,
                                        TECHNIQUE_BUCKETS, ranks_by_method))
        all_rows.extend(analyze_buckets(split, "group size", group_bucket_by_edge,
                                        GROUP_BUCKETS, ranks_by_method))
        print_cold_start(split)

    save_rows(all_rows, RESULT_FILE)


if __name__ == "__main__":
    main()