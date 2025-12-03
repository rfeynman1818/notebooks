# ================================================================
# 6. DriftLens execution (RUN AFTER SECTION 5)
# ================================================================
from driftlens.driftlens import DriftLens

dl = DriftLens()

# load saved embeddings
train_npz = np.load(train_path)
test_npz  = np.load(test_path)
stream_npz = np.load(stream_path)

E_train = train_npz["embeddings"]
Y_train_pred = train_npz["preds"]

E_test = test_npz["embeddings"]
Y_test_pred = test_npz["preds"]

E_stream = stream_npz["embeddings"]
Y_stream_pred = stream_npz["preds"]

# ---- 6.1 Baseline estimation ----
baseline = dl.estimate_baseline(
    E=E_train,
    Y=Y_train_pred,
    label_list=training_label_list_full,
    batch_n_pc=150,
    per_label_n_pc=75,
)

# ---- 6.2 Threshold estimation ----
per_batch_sorted, per_label_sorted = dl.random_sampling_threshold_estimation(
    label_list=training_label_list_full,
    E=E_test,
    Y=Y_test_pred,
    batch_n_pc=150,
    per_label_n_pc=75,
    window_size=1000,
    n_samples=1000,
    flag_shuffle=True,
    flag_replacement=True,
)

# ---- 6.3 Drift detection on blurred stream ----
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
