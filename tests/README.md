# Evaluation Scripts Moved

The evaluation scripts have been migrated to `src/`:

- `pose_dataset.py` → `src/pose_dataset.py`
- `precompute_embeddings.py` → `src/precompute_embeddings.py`
- `zero_shot.py` → `src/zero_shot.py`
- `few_shot.py` → `src/few_shot.py`

See [src/README.md](../src/README.md) for updated documentation.

The Colab notebook is still in `tests/collab_zero_shot.ipynb` but now references the new `src/` paths.
