from splits import NODE_FILE, EDGE_FILE, load_nodes, load_edges, make_time_split

CUTOFF_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


def get_training_node_ids(split):
    # Returns the set of node ids touched by at least one training edge.
    node_ids = set()
    for source_id, target_id in split["train_graph_pairs"]:
        node_ids.add(source_id)
        node_ids.add(target_id)
    return node_ids


def explain_cold_edges(split):
    # Counts cold-start edges by which endpoint the training graph has never seen.
    training_node_ids = get_training_node_ids(split)
    missing_group = 0
    missing_technique = 0
    missing_both = 0
    for group_id, technique_id in split["cold_pairs"]:
        group_unknown = group_id not in training_node_ids
        technique_unknown = technique_id not in training_node_ids
        if group_unknown and technique_unknown:
            missing_both += 1
        elif group_unknown:
            missing_group += 1
        else:
            # Every cold edge has at least one unknown endpoint, so this case is a new technique
            missing_technique += 1
    return missing_group, missing_technique, missing_both


def main():
    nodes = load_nodes(NODE_FILE)
    edges = load_edges(EDGE_FILE)
    print("cutoff  train  usable_test  cold  |  cold because: new_group  new_technique  both")
    for cutoff_year in CUTOFF_YEARS:
        split = make_time_split(nodes, edges, cutoff_year, 0)
        missing_group, missing_technique, missing_both = explain_cold_edges(split)
        print(cutoff_year, "  ",
              len(split["train_group_technique_pairs"]), "  ",
              len(split["test_pairs"]), "  ",
              len(split["cold_pairs"]), "  |  ",
              missing_group, "  ",
              missing_technique, "  ",
              missing_both)


if __name__ == "__main__":
    main()