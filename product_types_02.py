import collections
import matplotlib.pyplot as plt

tree_file = "tree.txt"

def parse_tree(tree_path):
    counts = collections.Counter()

    with open(tree_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.strip() == ".":
                continue

            idx_tee = max(line.rfind("├"), line.rfind("└"))
            if idx_tee == -1:
                continue

            name = line[idx_tee + 4:].strip()   # text after "├── " / "└── "

            if not name.endswith(".geojson"):
                continue

            parts = name.split("_")
            try:
                sicd_idx = parts.index("SICD")
                mode = parts[sicd_idx + 1]      # SLED, SLEDF, SLEDP, ...
            except (ValueError, IndexError):
                continue

            counts[mode] += 1

    return counts

mode_counts = parse_tree(tree_file)

modes, counts = zip(*sorted(mode_counts.items(), key=lambda x: x[0]))

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(modes, counts)
ax.set_title("Image Count By Mode")
ax.set_xlabel("Mode")
ax.set_ylabel("Image Count")
ax.set_xticks(range(len(modes)))
ax.set_xticklabels(modes, rotation=45, ha="right")
plt.tight_layout()
plt.show()
