# 5th ABAW Competition

## 1. Create environment
Create a python environment using conda, install packages in requirements.txt with `pip install -r requirements.txt`, or manually. 

## 2. Preprocess data and extract feature

- extract wavs from videos using `extract_wav.py`.
- extract visual features, audio features using pretrained models.
- construct samples dataset using `construct_*.py`.
- split samples dataset to segments using `split_*.py`.


## 3. Train model

Train model with `VA/solver.py`, `EXPR/solver.py` and `AU/solver.py`.

## 4. Predict

Predict with `VA/test.py`, `EXPR/test.py` and `AU/test.py`.