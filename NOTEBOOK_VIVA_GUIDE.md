# Notebook Viva Guide (MLPR)

## 1) Big Picture Pipeline

1. `merged_file.csv` and raw menu/sales sheets are prepared in EDA notebooks.
2. `may_till_aug_file.csv` and `may.csv` are derived datasets with engineered features.
3. `footfall_735.csv` is the main modeling dataset used by prediction notebooks and the app.
4. `models/poissonreg.pkl` is the trained model artifact used at runtime.
5. `model_demo.py` (and `models/model_demo.py`) serves the Gradio app for footfall prediction.

## 2) Runtime-Critical Files (for demo/app)

- `model_demo.py`: web app entrypoint.
- `models/poissonreg.pkl`: trained Poisson model loaded by app.
- `footfall_735.csv`: historical reference used for charts/comparisons in app.

## 3) Notebook-by-Notebook Explanation

### `clean.ipynb`
- Purpose: Merge footfall dataset with menu text features.
- Reads: `footfall_735.csv`, `merged_menu.csv`
- Writes: `menuxfootfall.csv`
- Connection: Feeds recommendation/deep-learning experiments in `reccomend.ipynb`.

### `concat.ipynb`
- Purpose: Restructure multiple weekly menu files and combine them.
- Reads: multiple menu `.xlsx/.csv` files, `merged_file.csv`, `horrid.csv`
- Writes: `menu_final.csv`, `horrid.csv`
- Connection: Intermediate data wrangling before menu-feature engineering.

### `eda/eda.ipynb`
- Purpose: Main exploratory analysis and feature engineering over footfall/sales data.
- Reads: `../merged_file.csv`, external August sales excel, `../may_till_aug_file.csv`
- Writes: `../may_till_aug_file.csv`, `../may.csv`
- Connection: Produces cleaned/engineered datasets used by downstream modeling notebooks.

### `eda.ipynb`
- Purpose: Alternate copy of EDA notebook with similar flow.
- Reads: `../merged_file.csv`, external August sheet path variant, `../may_till_aug_file.csv`
- Writes: `../may_till_aug_file.csv`, `../may.csv`
- Connection: Duplicate/variant of EDA workflow.

### `footfall_1000.ipynb`
- Purpose: Augmentation experiment using `SMOTE` on meal-level footfall data.
- Reads: `extra_dtaset.xlsx`
- Writes: none explicit (mostly in-memory experimentation)
- Connection: Experimental balancing pipeline; not used directly by app.

### `footfall_pred.ipynb`
- Purpose: Classical regression/GLM-based footfall modeling (statsmodels-centric flow).
- Reads: `../footfall_735.csv`
- Writes: `poissonreg.pkl`
- Connection: Trains the Poisson model artifact consumed by app.

### `footfall_prediction_models.ipynb`
- Purpose: Compare multiple regressors (Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, SVR).
- Reads: engineered period dataset (code points to `may_till_aug_file.csv`)
- Writes: none guaranteed
- Connection: Model benchmarking notebook.

### `footfall_prediction_models_mayonly.ipynb`
- Purpose: Same benchmarking idea restricted to May-focused feature subset.
- Reads: engineered dataset (code points to `may.csv`)
- Writes: none guaranteed
- Connection: Sensitivity study on month-specific data.

### `menu.ipynb`
- Purpose: Build merged meal text columns (`breakfast_merge`, `lunch_merge`, etc.).
- Reads: `mess.csv`
- Writes: `merged_menu.csv`
- Connection: Upstream for `clean.ipynb` and recommendation experiments.

### `merged.ipynb`
- Purpose: Inspect/clean merged menu-footfall table.
- Reads: `horrid.csv`
- Writes: none explicit
- Connection: Validation notebook for merged intermediate data.

### `models/footfall_pred.ipynb`
- Purpose: Duplicate of `footfall_pred.ipynb` kept inside `models/`.
- Reads: `../footfall_735.csv`
- Writes: `poissonreg.pkl`
- Connection: Same training objective; location suggests model-artifact-focused version.

### `models/proportion_pred.ipynb`
- Purpose: Predict meal proportion variables with multiple regressors.
- Reads: `../data/extra_dtaset.xlsx`
- Writes: none guaranteed
- Connection: Duplicate/organized copy of proportion modeling.

### `proportion_pred.ipynb`
- Purpose: Proportion prediction experiments (Linear/Ridge/Lasso/ElasticNet/Tree/RF/KNN/SVR).
- Reads: `../data/extra_dtaset.xlsx`
- Writes: none guaranteed
- Connection: Feature-level modeling; independent from app runtime.

### `reccomend.ipynb`
- Purpose: Menu-based feature expansion (one-hot/PCA) and neural recommendation-style regression.
- Reads: `menuxfootfall.csv`, `columns.csv`
- Writes: `columns.csv`, `pcad.csv`
- Connection: Experimental notebook using menu text-derived features.

### `regression.ipynb`
- Purpose: Broad regression comparison + hyperparameter tuning + feature importance + Poisson comparison.
- Reads: `footfall_735.csv`
- Writes: none guaranteed
- Connection: Core algorithm comparison notebook for viva discussion.

### `regression_analysis_footfall.ipynb`
- Purpose: Compact analysis notebook (single-cell quick study).
- Reads/Writes: minimal explicit I/O
- Connection: Supplemental analysis.

### `test.ipynb`
- Purpose: TensorFlow/Keras test notebook for DNN regression pipeline.
- Reads: `footfall_735.csv`
- Writes: model weights in some runs
- Connection: Experimental deep-learning baseline.

## 4) How Notebooks Connect (Dependency Map)

- Menu feature branch:
  - `menu.ipynb` -> `merged_menu.csv`
  - `clean.ipynb` + `footfall_735.csv` + `merged_menu.csv` -> `menuxfootfall.csv`
  - `reccomend.ipynb` consumes `menuxfootfall.csv`

- EDA/engineered-data branch:
  - `eda/eda.ipynb` or `eda.ipynb` -> `may_till_aug_file.csv`, `may.csv`
  - `footfall_prediction_models.ipynb` uses `may_till_aug_file.csv`
  - `footfall_prediction_models_mayonly.ipynb` uses `may.csv`
  - `proportion_pred.ipynb` / `models/proportion_pred.ipynb` use `data/extra_dtaset.xlsx`

- Main deployment branch:
  - `footfall_735.csv` -> `footfall_pred.ipynb` (or `models/footfall_pred.ipynb`) -> `models/poissonreg.pkl`
  - `model_demo.py` loads `models/poissonreg.pkl` and `footfall_735.csv`

## 5) Viva Talking Points (Quick)

- Why Poisson: footfall is count data; Poisson GLM is interpretable and suitable for non-negative counts.
- Why multiple notebooks: the repo stores both production-oriented and experimental tracks (classical ML, DL, menu-feature augmentation, proportion modeling).
- Why duplicates exist: some notebooks are mirrored in `models/` as artifact-focused copies.
- What is actually deployed: only the Gradio app + `poissonreg.pkl` + historical dataset for charts.

## 6) Suggested Viva Narrative (2 minutes)

1. Start from objective: predict cafeteria footfall to support inventory planning.
2. Explain data prep: EDA notebooks clean merged footfall/menu/sales data and engineer features.
3. Explain modeling: regression comparison notebooks benchmark algorithms; Poisson model selected for count prediction and interpretability.
4. Explain deployment: trained model saved as `poissonreg.pkl`, loaded by `model_demo.py` for interactive prediction.
5. Mention extensions: menu-aware modeling (`reccomend.ipynb`) and proportion models as future enhancement paths.
