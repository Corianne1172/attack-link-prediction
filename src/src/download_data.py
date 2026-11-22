import os
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"
OUTPUT_PATH = os.path.join("data", "raw", "enterprise-attack.json")


def download_attack_data(url, output_path):
    # Downloads the ATT&CK STIX file unless it is already on disk.
    if os.path.exists(output_path):
        print("File already exists:", output_path)
        return
    print("Downloading...")
    urllib.request.urlretrieve(url, output_path)
    size_in_mb = os.path.getsize(output_path) / (1024 * 1024)
    print("Saved", output_path, "(%.1f MB)" % size_in_mb)


if __name__ == "__main__":
    download_attack_data(DATA_URL, OUTPUT_PATH)