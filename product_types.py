import collections
import matplotlib.pyplot as plt

tree_file = "tree.txt"

def parse_tree(tree_path):
    counts = collections.Counter()
    # keeps the last directory name at each depth
    level_dir = {}

    with open(tree_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.strip() == ".":
                continue

            # find the connector character for this line (├ or └)
            idx_tee = max(line.rfind("├"), line.rfind("└"))
            if idx_tee == -1:
                continue

            level = idx_tee // 4                # 0 = top-level dir, 1 = subdir, ...
            name = line[idx_tee + 4:].strip()   # text after "├── " / "└── "

            if "." in name:                     # treat as file
                if name.endswith(".geojson"):
                    mode = level_dir.get(0)
                    if mode is not None:
                        counts[mode] += 1
            else:                               # it's a directory
                level_dir[level] = name

    return counts

mode_counts = parse_tree(tree_file)

# ----- plot -----
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
