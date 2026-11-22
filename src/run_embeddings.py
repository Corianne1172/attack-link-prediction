import os

from splits import NODE_FILE, EDGE_FILE, load_nodes, load_edges, build_all_splits
from embeddings import build_embedding_context, SCORE_FUNCTIONS
from evaluate import evaluate_methods
from reporting import save_rows, print_summary

RESULT_FILE = os.path.join("results", "embeddings.csv")
EMBEDDING_SEED = 0


def main():
    nodes = load_nodes(NODE_FILE)
    edges = load_edges(EDGE_FILE)
    all_rows = []
    for split in build_all_splits(nodes, edges):
        context = build_embedding_context(split, EMBEDDING_SEED)
        all_rows.extend(evaluate_methods(split, context, SCORE_FUNCTIONS))
    save_rows(all_rows, RESULT_FILE)
    print_summary(all_rows, list(SCORE_FUNCTIONS))


if __name__ == "__main__":
    main()