import pickle
import sys
import importlib


def ensure_numpy_pickle_compat() -> None:
    """Map numpy._core.* paths to numpy.core.* for cross-version pickle loading."""
    try:
        np_core = importlib.import_module("numpy.core")
        sys.modules.setdefault("numpy._core", np_core)
        for submodule in ("multiarray", "umath", "numeric", "_multiarray_umath"):
            try:
                target = importlib.import_module(f"numpy.core.{submodule}")
                sys.modules.setdefault(f"numpy._core.{submodule}", target)
            except ModuleNotFoundError:
                pass
    except Exception:
        pass

# Load all artifacts (works fine in Colab's numpy 2.x)
artifacts = {}
ensure_numpy_pickle_compat()
for name in ["model.pkl", "scaler.pkl", "fs5_features.pkl", "grade_means.pkl", "label_encoder.pkl"]:
    with open(f"artifacts/{name}", "rb") as f:
        artifacts[name] = pickle.load(f)

# Check scaler feature count vs fs5_features
print("Scaler n_features:", artifacts["scaler.pkl"].n_features_in_)
print("fs5_features count:", len(artifacts["fs5_features.pkl"]))
print("First 5 fs5:", artifacts["fs5_features.pkl"][:5])
print("Scaler mean first 5:", artifacts["scaler.pkl"].mean_[:5])

# Re-save with protocol 4 (numpy 1.x compatible)
import os
os.makedirs("artifacts_compat", exist_ok=True)
for name, obj in artifacts.items():
    with open(f"artifacts_compat/{name}", "wb") as f:
        pickle.dump(obj, f, protocol=4)
    print(f"Saved: {name}")