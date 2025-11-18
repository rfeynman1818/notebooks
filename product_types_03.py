import collections
import calendar
import matplotlib.pyplot as plt

tree_file = "tree.txt"

def month_to_season(m):
    if m in (12, 1, 2): return "Winter"
    if m in (3, 4, 5): return "Spring"
    if m in (6, 7, 8): return "Summer"
    return "Autumn"

def parse_tree(tree_path):
    mode_counts = collections.Counter()
    month_counts = collections.Counter()
    season_counts = collections.Counter()

    with open(tree_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.strip() == ".": 
                continue

            idx_tee = max(line.rfind("├"), line.rfind("└"))
            if idx_tee == -1: 
                continue

            name = line[idx_tee + 4:].strip()
            if not name.endswith(".geojson"): 
                continue

            parts = name.split("_")

            try:
                sicd_idx = parts.index("SICD")
                mode = parts[sicd_idx + 1]
            except (ValueError, IndexError):
                continue

            mode_counts[mode] += 1

            ts_part = parts[-1]                     # e.g. 20250327T183109.geojson
            ts_part = ts_part.split(".")[0]         # 20250327T183109
            date_str = ts_part.split("T")[0]        # 20250327

            if len(date_str) == 8 and date_str.isdigit():
                month_num = int(date_str[4:6])      # MM from YYYYMMDD
                if 1 <= month_num <= 12:
                    month_name = calendar.month_name[month_num]
                    month_counts[month_name] += 1
                    season = month_to_season(month_num)
                    season_counts[season] += 1

    return mode_counts, month_counts, season_counts

mode_counts, month_counts, season_counts = parse_tree(tree_file)

modes, mode_vals = zip(*sorted(mode_counts.items(), key=lambda x: x[0]))

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(modes, mode_vals)
ax.set_title("Image Count By Mode")
ax.set_xlabel("Mode")
ax.set_ylabel("Image Count")
ax.set_xticks(range(len(modes)))
ax.set_xticklabels(modes, rotation=45, ha="right")
plt.tight_layout()
plt.show()

all_months = list(calendar.month_name[1:])  # Jan..Dec
month_vals = [month_counts[m] for m in all_months]

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(all_months, month_vals)
ax.set_title("Image Count By Month Of Collection")
ax.set_xlabel("Month")
ax.set_ylabel("Image Count")
ax.set_xticks(range(len(all_months)))
ax.set_xticklabels(all_months, rotation=45, ha="right")
plt.tight_layout()
plt.show()

seasons = ["Winter", "Spring", "Summer", "Autumn"]
season_vals = [season_counts[s] for s in seasons]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(seasons, season_vals)
ax.set_title("Image Count By Season Of Collection")
ax.set_xlabel("Season")
ax.set_ylabel("Image Count")
plt.tight_layout()
plt.show()
