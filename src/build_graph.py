import csv
import json
import os

DATA_PATH = os.path.join("data", "raw", "enterprise-attack.json")
NODE_FILE = os.path.join("data", "processed", "nodes.csv")
EDGE_FILE = os.path.join("data", "processed", "edges.csv")

NODE_KIND_BY_TYPE = {
    "intrusion-set": "group",
    "attack-pattern": "technique",
    "malware": "software",
    "tool": "software",
}


def load_objects(file_path):
    # Loads the STIX bundle and returns its list of objects.
    with open(file_path, "r") as stix_file:
        bundle = json.load(stix_file)
    return bundle["objects"]


def is_active(stix_object):
    # Returns True if the object is not revoked or deprecated.
    if stix_object.get("revoked", False):
        return False
    if stix_object.get("x_mitre_deprecated", False):
        return False
    return True


def get_attack_id(stix_object):
    # Returns the ATT&CK ID of an object, such as T1059.001 or G0016.
    for reference in stix_object.get("external_references", []):
        if reference.get("source_name") == "mitre-attack":
            return reference.get("external_id")
    return ""


def collect_nodes(objects):
    # Returns a dictionary of node info keyed by STIX id for groups, techniques and software.
    nodes = {}
    for stix_object in objects:
        if not is_active(stix_object):
            continue
        object_type = stix_object["type"]
        if object_type not in NODE_KIND_BY_TYPE:
            continue
        nodes[stix_object["id"]] = {
            "attack_id": get_attack_id(stix_object),
            "name": stix_object["name"],
            "kind": NODE_KIND_BY_TYPE[object_type],
            "created": stix_object["created"],
        }
    return nodes


def get_edge_kind(source_kind, target_kind):
    # Names an edge by the kinds of its endpoints, or returns None if we do not use that pairing.
    if source_kind == "group" and target_kind == "technique":
        return "group_technique"
    if source_kind == "group" and target_kind == "software":
        return "group_software"
    if source_kind == "software" and target_kind == "technique":
        return "software_technique"
    return None


def collect_edges(objects, nodes):
    # Returns a dictionary of "uses" edges keyed by (source, target) with kind and earliest created date.
    edges = {}
    for stix_object in objects:
        if stix_object["type"] != "relationship":
            continue
        if stix_object["relationship_type"] != "uses":
            continue
        if not is_active(stix_object):
            continue
        source_id = stix_object["source_ref"]
        target_id = stix_object["target_ref"]
        if source_id not in nodes or target_id not in nodes:
            continue
        edge_kind = get_edge_kind(nodes[source_id]["kind"], nodes[target_id]["kind"])
        if edge_kind is None:
            continue
        key = (source_id, target_id)
        created = stix_object["created"]
        if key not in edges or created < edges[key]["created"]:
            edges[key] = {"kind": edge_kind, "created": created}
    return edges


def save_nodes(nodes, file_path):
    # Writes the nodes to a CSV file.
    with open(file_path, "w", newline="") as node_file:
        writer = csv.writer(node_file)
        writer.writerow(["node_id", "attack_id", "name", "kind", "created"])
        for node_id, info in nodes.items():
            writer.writerow([node_id, info["attack_id"], info["name"], info["kind"], info["created"]])


def save_edges(edges, file_path):
    # Writes the edges to a CSV file.
    with open(file_path, "w", newline="") as edge_file:
        writer = csv.writer(edge_file)
        writer.writerow(["source_id", "target_id", "kind", "created"])
        for (source_id, target_id), info in edges.items():
            writer.writerow([source_id, target_id, info["kind"], info["created"]])


def print_summary(nodes, edges):
    # Prints node counts, edge counts, and group-technique edges per created year.
    node_counts = {}
    for info in nodes.values():
        node_counts[info["kind"]] = node_counts.get(info["kind"], 0) + 1
    print("Nodes:")
    for kind in sorted(node_counts):
        print("  ", kind, node_counts[kind])

    edge_counts = {}
    year_counts = {}
    for info in edges.values():
        edge_counts[info["kind"]] = edge_counts.get(info["kind"], 0) + 1
        if info["kind"] == "group_technique":
            year = info["created"][:4]
            year_counts[year] = year_counts.get(year, 0) + 1
    print("Edges:")
    for kind in sorted(edge_counts):
        print("  ", kind, edge_counts[kind])

    print("Group-technique edges by year created:")
    for year in sorted(year_counts):
        print("  ", year, year_counts[year])


def main():
    objects = load_objects(DATA_PATH)
    nodes = collect_nodes(objects)
    edges = collect_edges(objects, nodes)
    save_nodes(nodes, NODE_FILE)
    save_edges(edges, EDGE_FILE)
    print_summary(nodes, edges)


if __name__ == "__main__":
    main()