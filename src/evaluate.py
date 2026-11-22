import random
from sklearn.metrics import roc_auc_score

HITS_K_VALUES = [10, 50]


def compute_auc(score_function, context, split):
    # AUC of the score function at separating held-out edges from negative pairs.
    positive_pairs = split["test_pairs"]
    negative_pairs = split["negative_pairs"]
    labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)
    scores = []
    for group_id, technique_id in positive_pairs + negative_pairs:
        scores.append(score_function(context, group_id, technique_id))
    return roc_auc_score(labels, scores)


def group_held_out_techniques(test_pairs):
    # Collects each group's held-out techniques into one set per group.
    held_out_by_group = {}
    for group_id, technique_id in test_pairs:
        held_out_by_group.setdefault(group_id, set()).add(technique_id)
    return held_out_by_group


def compute_hits(score_function, context, split, seed):
    # For each test group, the share of its held-out techniques ranked in the top k, averaged over groups.
    rng = random.Random(seed)
    held_out_by_group = group_held_out_techniques(split["test_pairs"])
    hit_totals = {}
    for k in HITS_K_VALUES:
        hit_totals[k] = 0.0

    for group_id, held_out in held_out_by_group.items():
        already_known = context["techniques_by_group"].get(group_id, set())
        ranked = []
        for technique_id in split["technique_ids"]:
            # Techniques the group already uses in training are not candidates
            if technique_id in already_known:
                continue
            score = score_function(context, group_id, technique_id)
            # The random number breaks ties fairly, so methods with many equal scores
            # are neither helped nor hurt by the order of the technique ids
            ranked.append((-score, rng.random(), technique_id))
        ranked.sort()
        for k in HITS_K_VALUES:
            top_k = set()
            for item in ranked[:k]:
                top_k.add(item[2])
            hit_totals[k] += len(top_k & held_out) / len(held_out)

    results = {}
    for k in HITS_K_VALUES:
        results[k] = hit_totals[k] / len(held_out_by_group)
    return results