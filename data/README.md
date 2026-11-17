Dataset: fedesoriano, Heart Failure Prediction Dataset (2021).
Original: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

Run `python scripts/download_data.py` from the repository root. If Kaggle requires login,
download the archive in your browser, extract heart.csv, and place it in this directory.
Expected: 918 rows, 12 columns, 508 positive and 410 negative labels.
No dataset is bundled or fabricated. The training loader validates these counts, schema,
category values, numeric ranges and duplicate rows. SHA-256 provenance is saved on training.
This predicts the dataset's HeartDisease label, not time-to-heart-failure or future incidence.
