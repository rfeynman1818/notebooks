## DriftLens Experiments on STL-10 (Use Cases 7.1 and 8)

This notebook sets up the environment and code needed to reproduce two DriftLens image experiments on the STL-10 dataset using a Vision Transformer classifier:

- **Use case 7.1 – Novel class drift (STL-10, ViT, F1 ≈ 0.96)**  
  - **Task:** 10-class image classification on STL-10.  
  - **Training labels:** Airplane, Bird, Car, Cat, Deer, Dog, Horse, Monkey, Ship  
    (the model is *not* trained on class **Truck**).  
  - **Drift scenario:** During the data stream, a new unseen class (**Truck**) begins to appear, simulating concept drift via a novel class.

- **Use case 8 – Input blur drift (STL-10, ViT, F1 ≈ 0.90)**  
  - **Task:** 10-class image classification on STL-10.  
  - **Training labels:** All ten classes are present: Airplane, Bird, Car, Cat, Deer, Dog, Horse, Monkey, Ship, Truck.  
  - **Drift scenario:** No new labels appear, but images in the stream are progressively corrupted by **Gaussian blur** for all classes, simulating data/quality drift.

DriftLens detects these drifts in an **unsupervised** way by monitoring how the distribution of model embeddings changes over time.

---

### Notebook setup

The first code cells configure the Jupyter environment:

- `%load_ext autoreload` and `%autoreload 2` enable **automatic reloading of local Python modules**. When you edit a module on disk (e.g., in `data/` or `utils/`), the updated version is picked up without restarting the kernel.
- `%matplotlib inline` makes all matplotlib figures appear directly inside the notebook.
- `nest_asyncio.apply()` patches the event loop used by Jupyter so that asynchronous code (if any) can be executed safely inside notebook cells.

These are purely **developer-experience** features; they do not affect the ML logic.

---

### Core libraries and utilities

The next cell imports general-purpose libraries:

- `os`, `json`, `copy`, `random`, `pathlib.Path` for filesystem and basic Python utilities.
- `numpy as np` for numerical operations and array manipulation.
- `torch` and `torchvision.transforms` for building and running deep models on images.
- `tqdm` for progress bars in training and embedding extraction loops.

Together these provide the basic scaffolding for dataset handling, model training, and metric computation.

---

### Reproducibility and device selection

We then fix random seeds and pick the computation device:

```python
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
