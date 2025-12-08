# role_of_datapreparation 🧹📊
**The Role of Data Preparation in Machine Learning**  
*(Python + Power BI data storytelling with NYC real-estate data, Nov 2024 → Oct 2025)*

---

## 📋 Table of Contents
- [Overview](#overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Tech Stack & System Requirements](#tech-stack--system-requirements)
- [Setup](#setup)
- [How to Run](#how-to-run)
  - [1) Notebook `main.ipynb`](#1-notebook-mainipynb)
  - [2) Script `visual.py`](#2-script-visualpy)
  - [3) Power BI Report (Data Storytelling)](#3-power-bi-report-data-storytelling)
- [ML Workflow & Data Storytelling](#ml-workflow--data-storytelling)
- [Expected Results (Example)](#expected-results-example)
- [Extensions](#extensions)
- [Author](#author)

---

## Overview
This project demonstrates the **impact of Data Preparation** on model performance in a real-world ML pipeline. Instead of focusing only on algorithms, it shows—through hands-on experiments—that:

> **Better data → better models. Data preparation determines performance.**

The core model is a **Decision Tree** implemented in Python.  
Additionally, the repository includes a **Power BI report** for end-to-end **data storytelling**, from data understanding to model comparison.

---

## Objectives
- Clearly show how **Data Preparation** affects model quality.
- Practice:
  - Cleaning (missing values, outliers, inconsistencies)
  - Feature transformation & encoding (scaling, one-hot/ordinal, etc.)
  - Feature selection / engineering
- Train a Decision Tree and **compare before vs after preparation**.
- Visualize the journey **raw → prepared → model → insights** in Python and **Power BI**.

---

## Dataset
- **Name:** `rolling_sale_data`
- **Scope:** New York City real-estate records from **Nov 2024** to **Oct 2025**
- **Suggested fields** (update to match your schema):
  - Location: `borough`, `neighborhood`, `address`, …
  - Property attributes: `building_class`, `land_sqft`, `gross_sqft`, `bedrooms`, `bathrooms`, …
  - Transaction: `sale_price`, `sale_date`, …
- **Format:** `csv` / `parquet` / `xlsx` (place under `data/`)

> Consider adding a separate schema document describing columns and dtypes.

---

## Repository Structure
```bash
role_of_datapreparation/
├── main.ipynb                    # Data prep + Decision Tree + evaluation
├── visual.py                     # Visuals: data understanding → model comparison
├── reports/
│   └── rolling_sales_story.pbix  # Power BI report for data storytelling
├── data/
│   └── rolling_sale_data.*       # NYC dataset (Nov 2024 – Oct 2025)
├── requirements.txt              # Optional: Python deps
└── README.md
Tech Stack & System Requirements
Python (recommended ≥ 3.9):

pandas, numpy

scikit-learn (Decision Tree & metrics)

matplotlib, seaborn

jupyter / jupyterlab

Power BI:

Power BI Desktop (Windows) to open .pbix.

Setup
Clone

bash
Sao chép mã
git clone https://github.com/<your-username>/role_of_datapreparation.git
cd role_of_datapreparation
(Recommended) Create a virtual environment

bash
Sao chép mã
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
Install dependencies

If you have requirements.txt:

bash
Sao chép mã
pip install -r requirements.txt
Or install the essentials:

bash
Sao chép mã
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
Sample requirements.txt (optional):

txt
Sao chép mã
pandas>=2.0
numpy>=1.23
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.13
jupyterlab>=4.0
How to Run
1) Notebook main.ipynb
Launch Jupyter and open the notebook:

bash
Sao chép mã
jupyter lab
# or
jupyter notebook
Follow the cells in order to:

Load data from data/rolling_sale_data.*

Data Understanding: distributions, correlations, missing, outliers

Data Preparation: impute/drop, outlier handling, encoding, scaling (if needed), train/test split

Modeling (Decision Tree): train with raw and prepared data

Evaluation: compare metrics before vs after (e.g., MAE/MSE/R² for regression; Accuracy/F1/AUC for classification)

Save any artifacts (plots, metrics, models) if implemented

2) Script visual.py
Purpose: consolidate Data Understanding & Model Comparison visuals.

Run:

bash
Sao chép mã
python visual.py
Expected outputs (depending on your implementation):

Feature distributions (hist/box/violin)

Missing-value heatmap

Outlier views (boxplot)

Model comparison charts (bar/line/table)

Feature importance

By default, the script reads from data/rolling_sale_data.*.
If you add CLI options (e.g., --data, --out), document them or provide --help.

3) Power BI Report (Data Storytelling)
Open reports/rolling_sales_story.pbix in Power BI Desktop, then point to your local dataset:

Home → Transform data → Data source settings

Select the current source → Change Source / Browse to data/rolling_sale_data.*

Apply changes → Refresh

Suggested report pages / visuals:

Overview: core KPIs (transaction count, median/avg price, by time)

Geography: by borough / neighborhood

Data Quality: missing/outlier rates by field/segment

Model Impact: Raw vs Prepared metrics & feature importance

Slicers/Filters: time (Nov 2024–Oct 2025), area, property type

ML Workflow & Data Storytelling
Data Understanding

Inspect schema/dtypes and distributions

Identify missing values & outliers

Explore relationships (corr, scatter, grouped summaries)

Data Preparation

Impute/drop missing, handle outliers (capping/winsorize, transforms)

Encode categoricals (one-hot/ordinal), scale features if needed

Split data (train/test or train/val/test)

Modeling — Decision Tree

Train two scenarios:

Raw data (minimal preparation)

Prepared data (cleaned/transformed)

Tune hyperparameters (e.g., max_depth, min_samples_split, min_samples_leaf)

Evaluation & Visualization

Compare metrics (Accuracy/F1/AUC or MAE/MSE/R²)

Plots: Pred vs Actual, Residuals (regression), Confusion Matrix (classification)

Feature importance (if applicable)

Story & Insights

Summarize improvements due to preparation

Key findings across time/area/property segments

Expected Results (Example)
Model Variant	Example Metric
Decision Tree (raw data)	Accuracy ≈ 0.60 / R² ≈ 0.55
Decision Tree (prepared data)	Accuracy ≈ 0.80 / R² ≈ 0.75

These numbers are illustrative—actual results depend on data quality/size, preprocessing, and hyperparameters.

Extensions
Additional models: Random Forest, Gradient Boosting, XGBoost, LightGBM

Advanced feature engineering (interactions, transforms, domain features)

Cross-validation and hyperparameter search (grid/random/bayesian)

Interactive dashboards with Streamlit or Dash

Pipeline automation (Makefile, DVC, Prefect/Airflow)

Author
Project: role_of_datapreparation

Topic: The Role of Data Preparation in Machine Learning

Contact: [Your Name] · [Email / GitHub / LinkedIn]

If you find this useful, please ⭐ the repo and feel free to open issues/PRs.
