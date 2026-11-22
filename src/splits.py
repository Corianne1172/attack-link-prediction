import csv
import os
import random

NODE_FILE = os.path.join("data", "processed", "nodes.csv")
EDGE_FILE = os.path.join("data", "processed", "edges.csv")
TEST_FRACTION = 0.2
# Edges created in this year or earlier are training data in the time split
CUTOFF_YEAR = 2024
RANDOM_SEEDS = [0, 1, 2, 3, 4]


def load_nodes(file_path):
    # Reads nodes.csv into a dictionary of rows keyed by node id.
    nodes = {}
    with open(file_path, "r") as node_file:
        reader = csv.DictReader(node_file)
        for row in reader:
            nodes[row["node_id"]] = row
    return nodes


def load_edges(file_path):
    # Reads edges.csv into a list of rows.
    edges = []
    with open(file_path, "r") as edge_file:
        reader = csv.DictReader(edge_file)
        for row in reader:
            edges.append(row)
    return edges


def get_year(created_text):
    # Returns the year from a timestamp like 2024-03-05T10:00:00.000Z.
    return int(created_text[:4])


def get_pairs(edges):
    # Turns edge rows into (source, target) pairs.
    pairs = []
    for edge in edges:
        pairs.append((edge["source_id"], edge["target_id"]))
    return pairs


def get_node_ids_of_kind(nodes, kind, allowed_ids=None):
    # Lists the ids of one kind of node, optionally limited to an allowed set.
    node_ids = []
    for node_id, row in nodes.items():
        if row["kind"] != kind:
            continue
        if allowed_ids is not None and node_id not in allowed_ids:
            continue
        node_ids.append(node_id)
    return sorted(node_ids)


def sample_negative_pairs(group_ids, technique_ids, known_pairs, count, seed):
    # Samples group-technique pairs that are not an edge anywhere in the full data.
    rng = random.Random(seed)
    # A set avoids sampling the same negative pair twice
    negative_pairs = set()
    while len(negative_pairs) < count:
        pair = (rng.choice(group_ids), rng.choice(technique_ids))
        # Check against ALL known edges (train and test), so a hidden real edge is never used as a negative
        if pair not in known_pairs:
            negative_pairs.add(pair)
    # Sorting makes the order the same on every run, because set order is not fixed
    return sorted(negative_pairs)


def make_split_dictionary(name, train_group_technique_edges, train_software_edges, test_edges,
                          cold_edges, group_ids, technique_ids, known_pairs, seed):
    # Packs one split into a dictionary, including negative pairs for AUC.
    # One negative pair per test edge keeps the AUC evaluation balanced
    negative_pairs = sample_negative_pairs(group_ids, technique_ids, known_pairs, len(test_edges), seed)
    return {
        "name": name,
        # Everything the model is allowed to see: training group-technique edges plus software edges
        "train_graph_pairs": get_pairs(train_group_technique_edges) + get_pairs(train_software_edges),
        "train_group_technique_pairs": get_pairs(train_group_technique_edges),
        "test_pairs": get_pairs(test_edges),
        "cold_pairs": get_pairs(cold_edges),
        "negative_pairs": negative_pairs,
        "group_ids": group_ids,
        "technique_ids": technique_ids,
    }


def make_random_split(nodes, edges, seed):
    # Holds out a random fraction of group-technique edges as the test set.
    group_technique_edges = []
    software_edges = []
    for edge in edges:
        if edge["kind"] == "group_technique":
            group_technique_edges.append(edge)
        else:
            software_edges.append(edge)

    # Only group-technique edges are hidden, because those are the edges we want to predict
    shuffled_edges = list(group_technique_edges)
    random.Random(seed).shuffle(shuffled_edges)
    test_count = int(len(shuffled_edges) * TEST_FRACTION)
    test_edges = shuffled_edges[:test_count]
    train_group_technique_edges = shuffled_edges[test_count:]

    group_ids = get_node_ids_of_kind(nodes, "group")
    technique_ids = get_node_ids_of_kind(nodes, "technique")
    # Negatives must avoid every real edge, including the hidden test edges
    known_pairs = set(get_pairs(group_technique_edges))
    # Software edges all stay in the training graph, since they are not what we predict
    return make_split_dictionary("random_seed" + str(seed), train_group_technique_edges,
                                 software_edges, test_edges, [], group_ids, technique_ids,
                                 known_pairs, seed)


def make_time_split(nodes, edges, cutoff_year, seed):
    # Trains on edges created up to the cutoff year and tests on edges created after it.
    train_group_technique_edges = []
    train_software_edges = []
    later_group_technique_edges = []
    all_group_technique_edges = []
    for edge in edges:
        year = get_year(edge["created"])
        if edge["kind"] == "group_technique":
            all_group_technique_edges.append(edge)
            if year <= cutoff_year:
                train_group_technique_edges.append(edge)
            else:
                later_group_technique_edges.append(edge)
        # Software edges are also filtered by date, otherwise future knowledge leaks into training
        elif year <= cutoff_year:
            train_software_edges.append(edge)

    # A node is "known" if at least one training edge touches it
    nodes_in_training = set()
    for edge in train_group_technique_edges + train_software_edges:
        nodes_in_training.add(edge["source_id"])
        nodes_in_training.add(edge["target_id"])

    # A later edge is a normal test edge only if the model has seen both endpoints.
    # Otherwise it is cold-start: the model has nothing to base a prediction on.
    test_edges = []
    cold_edges = []
    for edge in later_group_technique_edges:
        if edge["source_id"] in nodes_in_training and edge["target_id"] in nodes_in_training:
            test_edges.append(edge)
        else:
            cold_edges.append(edge)

    # Negatives and rankings only use nodes that exist at the cutoff
    group_ids = get_node_ids_of_kind(nodes, "group", nodes_in_training)
    technique_ids = get_node_ids_of_kind(nodes, "technique", nodes_in_training)
    known_pairs = set(get_pairs(all_group_technique_edges))
    return make_split_dictionary("time_" + str(cutoff_year), train_group_technique_edges,
                                 train_software_edges, test_edges, cold_edges, group_ids,
                                 technique_ids, known_pairs, seed)


def check_no_leakage(split):
    # Stops with an error if any test or negative pair appears in the training graph.
    train_pairs = set(split["train_graph_pairs"])
    # A test edge in the training graph would let the model see the answer, which makes results meaningless
    for pair in split["test_pairs"] + split["cold_pairs"] + split["negative_pairs"]:
        if pair in train_pairs:
            raise ValueError("Leakage in " + split["name"] + ": " + str(pair))


def print_split_summary(split):
    # Prints the size of each part of a split.
    print(split["name"])
    print("   train graph edges:", len(split["train_graph_pairs"]))
    print("   train group-technique edges:", len(split["train_group_technique_pairs"]))
    print("   test edges:", len(split["test_pairs"]))
    print("   cold-start edges:", len(split["cold_pairs"]))
    print("   negative pairs:", len(split["negative_pairs"]))
    print("   candidate groups:", len(split["group_ids"]))
    print("   candidate techniques:", len(split["technique_ids"]))


def main():
    nodes = load_nodes(NODE_FILE)
    edges = load_edges(EDGE_FILE)
    splits = []
    for seed in RANDOM_SEEDS:
        splits.append(make_random_split(nodes, edges, seed))
    splits.append(make_time_split(nodes, edges, CUTOFF_YEAR, 0))
    for split in splits:
        check_no_leakage(split)
        print_split_summary(split)
    print("No leakage found in any split.")


if __name__ == "__main__":
    main()