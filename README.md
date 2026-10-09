# Data Science with Python: Internship Portfolio

Work completed during the **Virtual Data Science with Python Trainee** internship at **YuvaIntern (Henry Harvin)**, Aug-Oct 2026.
Each week has its own folder with the Python code, generated figures/tables, and the full written report in `reports/`.

> Weeks 1-5 are in this repo. Week 6 (capstone) is being added.

| Week | Topic | Dataset | Key result |
|---|---|---|---|
| 1 | Data acquisition, cleaning & preprocessing | Titanic (891 x 12) | 891 x 16 clean, 0 missing, no rows dropped; 5-fold CV accuracy **80.25% -> 82.60%** vs naive "drop NA" |
| 2 | Exploratory data analysis & visualization | Titanic (cleaned) | 13 annotated plots + 2 aggregation tables; 1st-class women **97%** vs 3rd-class men **14%** survival |
| 3 | Unsupervised learning & clustering | UCI Wine (178 x 13) | K-Means (k=3) + Ward hierarchical; silhouette 0.285; **ARI 0.897** vs true cultivars |
| 4 | Supervised learning (classification) | Titanic (cleaned) | 6 models compared with 5-fold CV; tuned Gradient Boosting on held-out test: **81.6% acc, F1 0.740, ROC-AUC 0.862** |
| 5 | Deep learning (CNN, TensorFlow/Keras) | Fashion-MNIST (70k images) | 75k-param CNN with augmentation + BatchNorm + Dropout; **~90% test accuracy** (90.2% in the committed run), ablation vs MLP and plain CNN |

## Repository structure

```
.
├── README.md
├── requirements.txt
├── LICENSE
├── data/
│   ├── README.md
│   ├── raw/                       # titanic.csv, wine.csv, fashion-mnist/
│   └── processed/                 # titanic_cleaned.csv, data_dictionary.csv, sample_cleaned_head10.csv
├── scripts/download_data.py       # re-downloads the raw Titanic CSV
├── week1_data_cleaning/           # 01_explore.py, 02_clean_preprocess.py, 03_additional_analysis.py
├── week2_eda/                     # eda_analysis.py
├── week3_clustering/              # clustering_analysis.py
├── week4_supervised/              # model_pipeline.py
├── week5_deep_learning/           # data_loader.py, dl_pipeline.py
└── reports/                       # weekly .docx reports
```

Every `weekN_*/` folder writes plots to `figures/` and tables to `results/`.

## Setup

```bash
git clone <your-repo-url>
cd internship-data-science-python
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run order

Scripts can be run from any directory. Week 2 needs the cleaned file that Week 1 produces; Week 3 is standalone; Week 4 also needs the Week 1 output; Week 5 needs `data/raw/fashion-mnist/` (included, or run `python scripts/download_data.py`).

```bash
python week1_data_cleaning/01_explore.py            # profile raw data
python week1_data_cleaning/02_clean_preprocess.py   # writes data/processed/titanic_cleaned.csv
python week1_data_cleaning/03_additional_analysis.py
python week2_eda/eda_analysis.py
python week3_clustering/clustering_analysis.py      # uses scikit-learn's built-in wine data
python week4_supervised/model_pipeline.py           # needs data/processed/titanic_cleaned.csv (Week 1)
python week5_deep_learning/dl_pipeline.py           # TensorFlow; ~30 min on CPU
```

Random seeds are fixed (`random_state=42`), so the numbers above reproduce exactly.

## Methods at a glance

- **Week 1**: missingness diagnosed before fixing (chi-square test shows Age is MAR w.r.t. Pclass, so group-wise median imputation); `Cabin` turned into a `HasCabin` flag; 15 zero fares re-imputed; Fare outliers compared across Z-score / IQR(1.5) / IQR(3) and winsorised, not dropped; engineered `FamilySize`, `IsAlone`, `Title`. Data dictionary: `data/processed/data_dictionary.csv`.
- **Week 2**: univariate, bivariate and multivariate analysis with groupby aggregation tables; findings tied back to Week 1 preprocessing choices (e.g. Age capping removes the 60+ group).
- **Week 3**: features standardised; k picked with elbow + silhouette (peak at k=3); K-Means cross-checked with Ward hierarchical (169/178 samples agree); PCA used only for 2-D visualisation (55.4% variance); clusters profiled per feature and validated externally against the cultivar labels, which were never used for fitting.
- **Week 4**: stratified 80/20 split (712/179), scaler fit on train only (no leakage), 6 algorithms scored on 5 metrics via stratified 5-fold CV, GridSearchCV on Random Forest and Gradient Boosting, final test evaluation with confusion matrix, ROC curve, feature importance and learning curve. Top predictors: Title, Pclass, Sex and an Age x Class interaction, which matches the Week 2 EDA.
- **Week 5**: Fashion-MNIST split 54k train / 6k validation / 10k test; CNN (4 conv layers, BatchNorm, Dropout, GlobalAveragePooling) with random translation/zoom/flip augmentation, Adam + EarlyStopping + ReduceLROnPlateau; per-class precision/recall/F1, confusion matrix, misclassified samples, and an ablation study. Shirt is the weakest class (confused with T-shirt, Pullover, Coat), which is a known property of Fashion-MNIST at 28x28. Note: TensorFlow on CPU with random augmentation is not bit-reproducible even with fixed seeds, so a re-run lands within about 1 point of the committed outputs (committed run: 90.24% test accuracy, MLP baseline 84.1%, unregularised CNN 69.8%). The `.docx` report quotes an earlier run (89.25%); the conclusions are the same.

### Limitations / future work (Week 3)

Silhouette of ~0.28 means clusters overlap somewhat, which is expected since chemically similar wines exist across cultivars. With only 178 samples from one region, boundaries may not generalise. Next steps would be trying Gaussian Mixture Models (soft assignment), DBSCAN, and a stability check across random seeds.

### Limitations / future work (Weeks 4-5)

Week 4: only 891 rows, so test metrics (179 passengers) are noisy; recall (68%) is lower than precision, so a threshold shift or class weights could be tried. Week 5: trained on CPU with a 12-epoch cap; a deeper network or longer schedule might lift the upper-body garment classes slightly, but much of the Shirt confusion is a data limitation.

## Reports

[`reports/`](reports/) has the full write-ups (methodology, figures, interpretation, limitations).

## Author

**Aman**: B.Tech Software Engineering, NIAT Jaipur · [LinkedIn](https://www.linkedin.com/in/aman-kumar51)

Released under the MIT License.
