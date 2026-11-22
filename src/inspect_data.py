import json
import os

DATA_PATH = os.path.join("data", "raw", "enterprise-attack.json")


def load_objects(file_path):
    # Loads the STIX bundle and returns its list of objects.
    with open(file_path, "r") as stix_file:
        bundle = json.load(stix_file)
    return bundle["objects"]


def count_by_type(objects):
    # Counts how many objects there are of each STIX type.
    counts = {}
    for stix_object in objects:
        object_type = stix_object["type"]
        counts[object_type] = counts.get(object_type, 0) + 1
    return counts


def count_inactive(objects):
    # Counts objects that are revoked or deprecated.
    inactive_count = 0
    for stix_object in objects:
        if stix_object.get("revoked", False) or stix_object.get("x_mitre_deprecated", False):
            inactive_count += 1
    return inactive_count


def print_attack_version(objects):
    # Prints the ATT&CK collection name and version, which the report must cite.
    for stix_object in objects:
        if stix_object["type"] == "x-mitre-collection":
            print("Collection:", stix_object.get("name"))
            print("ATT&CK version:", stix_object.get("x_mitre_version"))
            print("Modified:", stix_object.get("modified"))


def main():
    objects = load_objects(DATA_PATH)
    print("Total objects:", len(objects))
    print_attack_version(objects)
    print()
    counts = count_by_type(objects)
    for object_type in sorted(counts, key=counts.get, reverse=True):
        print(object_type, counts[object_type])
    print()
    print("Revoked or deprecated objects:", count_inactive(objects))


if __name__ == "__main__":
    main()