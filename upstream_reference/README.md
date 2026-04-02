# Upstream SignCLIP Reference Scripts

This folder contains **original scripts from the SignCLIP paper's authors** (Jiang et al., 2024), copied from `fairseq/examples/MMPT` for reference purposes.

⚠️ **These are NOT our work** - they are provided as reference implementations from the original paper.

## Structure

### `demos/`
Original demonstration scripts:
- `demo.py` - General demo
- `demo_feature.py` - Feature extraction demo
- `demo_finger.py` - Fingerspelling demo

### `tests/`
Original evaluation scripts:
- `test_iconicity.py` - Iconicity testing
- `test_identification_dicta.py` - Identification tests
- `test_recognition_few_shot.py` - Few-shot recognition
- `test_recognition_few_shot_knn.py` - KNN-based few-shot
- `test_recognition_few_shot_external.py` - External dataset tests
- `test_recognition_supervised.py` - Supervised recognition

### `analysis/`
Original paper analysis and results scripts:
- `results_paper.py` - Main paper results
- `results_sign_clip.py` - SignCLIP results
- `results_sign_clip_sp.py` - SignCLIP spatial results
- `results_asl_signs.py` - ASL dataset results
- `results_rwthfs.py` - RWTH-FS dataset results
- `data_stat_asl.py` - ASL dataset statistics
- `data_stat_sp.py` - Spatial dataset statistics
- `extract_examples.py` - Extract example cases
- `extract_examples_iconicity.py` - Extract iconicity examples

## Citation

If you use these scripts, please cite the original SignCLIP paper:

```bibtex
@inproceedings{jiang-etal-2024-signclip,
    title = "{S}ign{CLIP}: Connecting Text and Sign Language by Contrastive Learning",
    author = {Jiang, Zifan and Sant, Gerard and Moryossef, Amit and 
              M{\"u}ller, Mathias and Sennrich, Rico and Ebling, Sarah},
    booktitle = "Proceedings of EMNLP 2024",
    year = "2024",
}
```

## Note

These scripts may have dependencies on the original fairseq-based MMPT setup. For our standalone experiments, see the main `scripts/` folder in the parent directory.
