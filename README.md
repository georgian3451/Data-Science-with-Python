# Data Science with Python: Internship Portfolio

Work completed during the **Virtual Data Science with Python Trainee** internship at **YuvaIntern (Henry Harvin)**, Aug-Oct 2026.
Each week has its own folder with the Python code, generated figures/tables, and the full written report in `reports/`.

> Weeks 1-2 are in this repo. Weeks 3-6 (clustering, supervised learning, deep learning, capstone) are being added.

| Week | Topic | Dataset | Key result |
|---|---|---|---|
| 1 | Data acquisition, cleaning & preprocessing | Titanic (891 x 12) | 891 x 16 clean, 0 missing, no rows dropped; 5-fold CV accuracy **80.25% -> 82.60%** vs naive "drop NA" |
| 2 | Exploratory data analysis & visualization | Titanic (cleaned) | 13 annotated plots + 2 aggregation tables; 1st-class women **97%** vs 3rd-class men **14%** survival |

## Repository structure

```
.
├── README.md
├── requirements.txt
├── LICENSE
├── data/
│   ├── README.md
│   ├── raw/titanic.csv
│   └── processed/                 # titanic_cleaned.csv, data_dictionary.csv, sample_cleaned_head10.csv
├── scripts/download_data.py       # re-downloads the raw Titanic CSV
├── week1_data_cleaning/           # 01_explore.py, 02_clean_preprocess.py, 03_additional_analysis.py
├── week2_eda/                     # eda_analysis.py
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

Scripts can be run from any directory. Week 2 needs the cleaned file that Week 1 produces.

```bash
python week1_data_cleaning/01_explore.py            # profile raw data
python week1_data_cleaning/02_clean_preprocess.py   # writes data/processed/titanic_cleaned.csv
python week1_data_cleaning/03_additional_analysis.py
python week2_eda/eda_analysis.py
```

Random seeds are fixed (`random_state=42`), so the numbers above reproduce exactly.

## Methods at a glance

- **Week 1**: missingness diagnosed before fixing (chi-square test shows Age is MAR w.r.t. Pclass, so group-wise median imputation); `Cabin` turned into a `HasCabin` flag; 15 zero fares re-imputed; Fare outliers compared across Z-score / IQR(1.5) / IQR(3) and winsorised, not dropped; engineered `FamilySize`, `IsAlone`, `Title`. Data dictionary: `data/processed/data_dictionary.csv`.
- **Week 2**: univariate, bivariate and multivariate analysis with groupby aggregation tables; findings tied back to Week 1 preprocessing choices (e.g. Age capping removes the 60+ group).

## Reports

[`reports/`](reports/) has the full write-ups (methodology, figures, interpretation, limitations).

## Author

**Aman**: B.Tech Software Engineering, NIAT Jaipur · [LinkedIn](https://www.linkedin.com/in/aman-kumar51)

Released under the MIT License.
