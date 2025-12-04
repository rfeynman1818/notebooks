# ================================================================
# 6. DriftLens execution (RUN AFTER SECTION 5)
# ================================================================
from driftlens.driftlens import DriftLens
import numpy as np

dl = DriftLens()

# ---- 6.0 Load saved embeddings ----
train_npz = np.load(train_path)
test_npz  = np.load(test_path)
stream_npz = np.load(stream_path)

E_train = train_npz["embeddings"]
Y_train_pred = train_npz["preds"]

E_test = test_npz["embeddings"]
Y_test_pred = test_npz["preds"]

E_stream = stream_npz["embeddings"]
Y_stream_pred = stream_npz["preds"]

print("Shapes:",
      "E_train", E_train.shape,
      "Y_train_pred", Y_train_pred.shape,
      "E_test", E_test.shape,
      "Y_test_pred", Y_test_pred.shape)

# For reference: class names and numeric ids
class_names = training_label_list_full                 # e.g. ["High", "Low", ...]
label_ids_all = list(range(len(class_names)))          # e.g. [0,1,2,3,4,5,6]

# ================================================================
# 6.1 Baseline estimation  (per-batch PCA only)
# ================================================================
# Empty list => no per-label PCA, only global/batch PCA
label_list_for_baseline = []

baseline = dl.estimate_baseline(
    E=E_train,
    Y=Y_train_pred,
    label_list=label_list_for_baseline,
    batch_n_pc=None,
    per_label_n_pc=None,
)

# ================================================================
# 6.2 Threshold estimation  (proportional sampling)
# ================================================================
# Here DriftLens expects label ids, not names
label_list_for_threshold = label_ids_all

# --- build proportions_dict from Y_test_pred (which uses numeric ids) ---
label_ids, counts = np.unique(Y_test_pred, return_counts=True)
total = counts.sum()

proportions_dict = {str(l): 0.0 for l in label_list_for_threshold}
for lab_id, c in zip(label_ids, counts):
    lab_id = int(lab_id)
    if lab_id in label_list_for_threshold:
        proportions_dict[str(lab_id)] = float(c) / float(total)

print("Label proportions used for threshold estimation:")
for i, name in enumerate(class_names):
    print(f"  id={i} name={name} prop={proportions_dict[str(i)]:.4f}")

per_batch_sorted, per_label_sorted = dl.random_sampling_threshold_estimation(
    label_list=label_list_for_threshold,   # numeric ids
    E=E_test,
    Y=Y_test_pred,
    batch_n_pc=None,
    per_label_n_pc=None,
    window_size=1000,
    n_samples=1000,
    flag_shuffle=True,
    flag_replacement=True,
    proportional_flag=True,
    proportions_dict=proportions_dict,
)

# ================================================================
# 6.3 Drift detection on blurred stream
# ================================================================
window_size = 1000
stride = 500

drift_scores = []
for start in range(0, len(E_stream) - window_size + 1, stride):
    E_win = E_stream[start:start+window_size]
    Y_win = Y_stream_pred[start:start+window_size]

    dist = dl.compute_window_distribution_distances(
        E=E_win,
        Y=Y_win,
        baseline=baseline,
        batch_sorted=per_batch_sorted,
        per_label_sorted=per_label_sorted,
    )

    drift_scores.append(dist)

print("Drift detection finished. Num windows:", len(drift_scores))
