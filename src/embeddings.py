import random
import zlib

import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression

from heuristics import build_scoring_context
from splits import sample_negative_pairs

# Embedding settings. These started as guesses and get updated after tuning on validation data.
EMBEDDING_SETTINGS = {
    "dimension": 64,
    "walks_per_node": 10,
    "walk_length": 80,
    "window": 40,
    "epochs": 10,
    "use_software": True,
}


def stable_hash(text):
    # Gives gensim a hash that is the same on every run (Python's built-in string hash changes per run).
    return zlib.crc32(text.encode("utf-8"))


def generate_random_walks(neighbors, walks_per_node, walk_length, seed):
    # Runs uniform random walks that start from every node, for several rounds.
    rng = random.Random(seed)
    # Sorted lists keep the walks identical on every run, because set order is not fixed
    neighbor_lists = {}
    for node_id, neighbor_set in neighbors.items():
        neighbor_lists[node_id] = sorted(neighbor_set)
    node_ids = sorted(neighbor_lists)

    walks = []
    for round_number in range(walks_per_node):
        # DeepWalk reshuffles the start order each round
        rng.shuffle(node_ids)
        for start_node in node_ids:
            walk = [start_node]
            while len(walk) < walk_length:
                current_node = walk[-1]
                walk.append(rng.choice(neighbor_lists[current_node]))
            walks.append(walk)
    return walks


def train_embeddings(walks, seed, settings):
    # Trains skip-gram on the walks. Returns each node's vector, its unit-length version,
    # and its "context" vector (the second set of vectors skip-gram learns).
    # Each walk plays the role of a sentence and each node the role of a word
    model = Word2Vec(
        sentences=walks,
        vector_size=settings["dimension"],
        window=settings["window"],
        min_count=1,
        sg=1,
        workers=1,
        epochs=settings["epochs"],
        seed=seed,
        hashfxn=stable_hash,
    )
    vectors = {}
    unit_vectors = {}
    output_vectors = {}
    for index, node_id in enumerate(model.wv.index_to_key):
        vector = model.wv.vectors[index]
        vectors[node_id] = vector
        unit_vectors[node_id] = vector / np.linalg.norm(vector)
        # syn1neg holds the context vectors, in the same row order as the node vectors
        output_vectors[node_id] = model.syn1neg[index]
    return vectors, unit_vectors, output_vectors


def train_link_classifier(context, split, seed):
    # Fits a logistic regression on training edges versus random non-edges.
    vectors = context["vectors"]
    train_pairs = split["train_group_technique_pairs"]
    positive_pairs = []
    for group_id, technique_id in train_pairs:
        if group_id in vectors and technique_id in vectors:
            positive_pairs.append((group_id, technique_id))

    # Same number of negatives as positives. Some sampled negatives may be real edges that are
    # hidden in the test set, which is unavoidable and affects very few pairs.
    negative_pairs = sample_negative_pairs(split["group_ids"], split["technique_ids"],
                                           set(train_pairs), len(positive_pairs), seed)

    features = []
    labels = []
    for group_id, technique_id in positive_pairs:
        # Multiplying the two vectors elementwise gives one feature vector for the pair
        features.append(vectors[group_id] * vectors[technique_id])
        labels.append(1)
    for group_id, technique_id in negative_pairs:
        if group_id in vectors and technique_id in vectors:
            features.append(vectors[group_id] * vectors[technique_id])
            labels.append(0)

    model = LogisticRegression(max_iter=1000)
    model.fit(np.array(features), np.array(labels))
    return model.coef_[0], model.intercept_[0]

def get_walk_neighbors(context, split, settings):
    # Chooses which graph the random walks run on: the full graph, or group-technique edges only.
    if settings["use_software"]:
        return context["neighbors"]
    neighbors = {}
    for group_id, technique_id in split["train_group_technique_pairs"]:
        neighbors.setdefault(group_id, set()).add(technique_id)
        neighbors.setdefault(technique_id, set()).add(group_id)
    return neighbors

def build_embedding_context(split, seed, settings=None):
    # Trains DeepWalk and the link classifier on the training graph only.
    if settings is None:
        settings = EMBEDDING_SETTINGS
    context = build_scoring_context(split)
    walk_neighbors = get_walk_neighbors(context, split, settings)
    walks = generate_random_walks(walk_neighbors, settings["walks_per_node"],
                                  settings["walk_length"], seed)
    vectors, unit_vectors, output_vectors = train_embeddings(walks, seed, settings)
    context["vectors"] = vectors
    context["unit_vectors"] = unit_vectors
    context["output_vectors"] = output_vectors
    weights, bias = train_link_classifier(context, split, seed)
    context["classifier_weights"] = weights
    context["classifier_bias"] = bias
    return context


def deepwalk_cosine_score(context, group_id, technique_id):
    # Cosine similarity of the group and technique vectors.
    unit_vectors = context["unit_vectors"]
    # A node with no training edges has no vector, so it gets a neutral score of 0
    if group_id not in unit_vectors or technique_id not in unit_vectors:
        return 0.0
    return float(np.dot(unit_vectors[group_id], unit_vectors[technique_id]))


def deepwalk_context_score(context, group_id, technique_id):
    # Skip-gram's own score: how likely each node is to show up near the other in a walk, averaged both ways.
    vectors = context["vectors"]
    output_vectors = context["output_vectors"]
    if group_id not in vectors or technique_id not in vectors:
        return 0.0
    group_to_technique = np.dot(vectors[group_id], output_vectors[technique_id])
    technique_to_group = np.dot(vectors[technique_id], output_vectors[group_id])
    return float((group_to_technique + technique_to_group) / 2)


def deepwalk_classifier_score(context, group_id, technique_id):
    # The classifier's log-odds that the group uses the technique.
    vectors = context["vectors"]
    if group_id not in vectors or technique_id not in vectors:
        return float(context["classifier_bias"])
    # Weights times (group vector * technique vector) is the same as (weights * group vector) . technique vector
    weighted_group = context["classifier_weights"] * vectors[group_id]
    return float(np.dot(weighted_group, vectors[technique_id])) + float(context["classifier_bias"])


SCORE_FUNCTIONS = {
    "deepwalk_cosine": deepwalk_cosine_score,
    "deepwalk_context": deepwalk_context_score,
    "deepwalk_classifier": deepwalk_classifier_score,
}