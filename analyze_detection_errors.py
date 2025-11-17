import json
import collections

# Hardcoded label map (example)
# Provide your actual mapping here:
CLASS_MAP = {
    0: "background",
    1: "person",
    2: "car",
    3: "dog",
    4: "cat",
}

with open("/path/to/detection_matching.json") as f:
    data = json.load(f)

def image_stat_dict():
    return {"confused": 0, "fp": 0, "fn": 0}

# label → filename → stats
label_image_stats = collections.defaultdict(lambda: collections.defaultdict(image_stat_dict))

# label → filename → matched_count
matched_counts = collections.defaultdict(lambda: collections.defaultdict(int))

# ------------------------------
#   AGGREGATE: Confused
# ------------------------------
for item in data.get("confused_detections", []):
    ann = item["annotation"]
    det = item["detect"]

    label_id = ann["label"]
    label = CLASS_MAP.get(label_id, f"unknown_{label_id}")
    filename = det["image"].split("/")[-1]

    label_image_stats[label][filename]["confused"] += 1

# ------------------------------
#   AGGREGATE: False Positives
# ------------------------------
for item in data.get("unmatched_detections", []):
    label_id = item["label"]
    label = CLASS_MAP.get(label_id, f"unknown_{label_id}")
    filename = item["image"].split("/")[-1]

    label_image_stats[label][filename]["fp"] += 1

# ------------------------------
#   AGGREGATE: False Negatives
# ------------------------------
for item in data.get("unmatched_annotations", []):
    label_id = item["label"]
    label = CLASS_MAP.get(label_id, f"unknown_{label_id}")
    filename = item["image"].split("/")[-1]

    label_image_stats[label][filename]["fn"] += 1

# ------------------------------
#   AGGREGATE: Matched
# ------------------------------
for item in data.get("matched_detections", []):
    det_label_id = item["detect"]["label"]
    label = CLASS_MAP.get(det_label_id, f"unknown_{det_label_id}")
    filename = item["annotation"]["image"].split("/")[-1]
    matched_counts[label][filename] += 1

# ------------------------------
#   OUTPUT
# ------------------------------
print("\n=== IMAGE ERRORS BY LABEL ===")

for label in CLASS_MAP.values():
    image_stats = label_image_stats.get(label, {})

    sorted_stats = sorted(
        image_stats.items(),
        key=lambda s: sum(s[1].values()),
        reverse=True
    )

    print(f"\nLabel: {label}")

    if not any(sum(v.values()) > 0 for v in image_stats.values()):
        print("No errors found.")
        continue

    for filename, counts in sorted_stats:
        if sum(counts.values()) == 0:
            continue
        print(f"\nImage: {filename}")
        print(f" - Confused: {counts['confused']}")
        print(f" - False Positives: {counts['fp']}")
        print(f" - False Negatives: {counts['fn']}")
        print("-" * 25)

# ------------------------------
#   OUTPUT MATCHED COUNTS
# ------------------------------
print("\n=== MATCHED ANNOTATIONS ===")

for label in CLASS_MAP.values():
    image_counts = matched_counts.get(label, {})
    if not image_counts:
        continue

    print(f"\nMatched for label: {label}")
    sorted_counts = sorted(image_counts.items(), key=lambda s: s[1], reverse=True)

    for filename, count in sorted_counts:
        print(f"{filename}: {count}")
