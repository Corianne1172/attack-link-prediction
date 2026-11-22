import math


def build_scoring_context(split):
    # Builds lookup tables from the training graph only, so scores never see test edges.
    # neighbors: every node's neighbors across all edge kinds (used by the classic heuristics)
    neighbors = {}
    for source_id, target_id in split["train_graph_pairs"]:
        neighbors.setdefault(source_id, set()).add(target_id)
        neighbors.setdefault(target_id, set()).add(source_id)

    # The group-technique edges alone are what the popularity and similar-group scores use
    techniques_by_group = {}
    groups_by_technique = {}
    for group_id, technique_id in split["train_group_technique_pairs"]:
        techniques_by_group.setdefault(group_id, set()).add(technique_id)
        groups_by_technique.setdefault(technique_id, set()).add(group_id)

    return {
        "neighbors": neighbors,
        "techniques_by_group": techniques_by_group,
        "groups_by_technique": groups_by_technique,
        "similarity_cache": {},
    }


def popularity_score(context, group_id, technique_id):
    # Scores a technique by how many training groups use it, ignoring which group is asking.
    return len(context["groups_by_technique"].get(technique_id, set()))


def common_neighbors_score(context, group_id, technique_id):
    # Counts the neighbors the group and technique share. Only software can be shared, since
    # groups connect to techniques and software, while techniques connect to groups and software.
    neighbors = context["neighbors"]
    group_neighbors = neighbors.get(group_id, set())
    technique_neighbors = neighbors.get(technique_id, set())
    return len(group_neighbors & technique_neighbors)


def jaccard_score(context, group_id, technique_id):
    # Shared neighbors divided by the size of the combined neighborhood.
    neighbors = context["neighbors"]
    group_neighbors = neighbors.get(group_id, set())
    technique_neighbors = neighbors.get(technique_id, set())
    combined = group_neighbors | technique_neighbors
    if len(combined) == 0:
        return 0.0
    return len(group_neighbors & technique_neighbors) / len(combined)


def adamic_adar_score(context, group_id, technique_id):
    # Shared neighbors weighted by 1 / log(degree), so rare shared neighbors count more.
    neighbors = context["neighbors"]
    group_neighbors = neighbors.get(group_id, set())
    technique_neighbors = neighbors.get(technique_id, set())
    total = 0.0
    for shared_node in group_neighbors & technique_neighbors:
        # A shared node touches both ends, so its degree is at least 2 and the log is above 0
        total += 1.0 / math.log(len(neighbors[shared_node]))
    return total


def get_group_similarity(context, group_a, group_b):
    # Cosine similarity of two groups' technique sets. Results are cached because they repeat a lot.
    if group_a < group_b:
        key = (group_a, group_b)
    else:
        key = (group_b, group_a)
    cache = context["similarity_cache"]
    if key not in cache:
        techniques_a = context["techniques_by_group"].get(group_a, set())
        techniques_b = context["techniques_by_group"].get(group_b, set())
        if len(techniques_a) == 0 or len(techniques_b) == 0:
            cache[key] = 0.0
        else:
            overlap = len(techniques_a & techniques_b)
            cache[key] = overlap / math.sqrt(len(techniques_a) * len(techniques_b))
    return cache[key]


def three_hop_paths_score(context, group_id, technique_id):
    # Counts paths group -> technique -> other group -> candidate technique.
    # Each other group that uses the candidate adds the number of techniques it shares with this group.
    own_techniques = context["techniques_by_group"].get(group_id, set())
    total = 0
    for other_group in context["groups_by_technique"].get(technique_id, set()):
        if other_group == group_id:
            continue
        other_techniques = context["techniques_by_group"].get(other_group, set())
        total += len(own_techniques & other_techniques)
    return total


def similar_groups_score(context, group_id, technique_id):
    # Adds up the similarity of this group to every other group that uses the candidate technique.
    # Unlike the raw path count, similarity is normalized, so huge groups do not dominate.
    total = 0.0
    for other_group in context["groups_by_technique"].get(technique_id, set()):
        if other_group == group_id:
            continue
        total += get_group_similarity(context, group_id, other_group)
    return total


SCORE_FUNCTIONS = {
    "popularity": popularity_score,
    "common_neighbors": common_neighbors_score,
    "jaccard": jaccard_score,
    "adamic_adar": adamic_adar_score,
    "three_hop_paths": three_hop_paths_score,
    "similar_groups": similar_groups_score,
}