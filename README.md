# role_of_datapreparation 🧹📊
**The Role of Data Preparation in Machine Learning**  
*(Python + Power BI data storytelling with NYC real-estate data, Nov 2024 → Oct 2025)*
<!-- Badges -->
![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![scikit-learn ≥1.3](https://img.shields.io/badge/scikit--learn-%E2%89%A51.3-F7931E?logo=scikit-learn&logoColor=white)
![pandas ≥2.0](https://img.shields.io/badge/pandas-%E2%89%A52.0-150458?logo=pandas&logoColor=white)
![seaborn](https://img.shields.io/badge/seaborn-0.13-9ECAE1?logo=python&logoColor=white)
![matplotlib](https://img.shields.io/badge/matplotlib-3.7-11557C?logo=python&logoColor=white)
![Focus: Data Preparation](https://img.shields.io/badge/Focus-Data%20Preparation-brightgreen)
![EDA](https://img.shields.io/badge/EDA-Exploration-blue)
![Model: Decision Tree](https://img.shields.io/badge/Model-Decision%20Tree-orange)
![Power BI](https://img.shields.io/badge/Power%20BI-Report-F2C811?logo=powerbi&logoColor=black)
![Dataset: NYC Real Estate](https://img.shields.io/badge/Dataset-NYC%20Real%20Estate-0A66C2)
![Time Range](https://img.shields.io/badge/Time%20Range-11%2F2024%E2%86%9210%2F2025-informational)
![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen)

---
## 🏷️ Tags

`machine-learning` · `data-preparation` · `data-cleaning` · `feature-engineering` · `exploratory-data-analysis` · `eda` · `model-evaluation` · `feature-importance` · `decision-tree` · `scikit-learn` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `jupyter-notebook` · `power-bi` · `data-visualization` · `data-storytelling` · `real-estate` · `nyc` · `housing-prices` · `tabular-data` · `supervised-learning` · `regression` · `classification` · `train-test-split` · `encoding` · `scaling` · `outliers` · `missing-data`
---
## 📋 Table of Contents
- [Overview](#overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Tech Stack & System Requirements](#tech-stack--system-requirements)
- [Setup](#setup)
- [How to Run](#how-to-run)
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
- **Format:** `xlsx` (place under `data/`)

> Consider adding a separate schema document describing columns and dtypes.

---

## Repository Structure
``` bash
role_of_datapreparation/
├── main.ipynb                    # Data prep + Decision Tree + evaluation
├── visual.py                     # Visuals: data understanding → model comparison
├── reports/
│   ├── rolling_sales_story.pbix  # Power BI report for data storytelling
│   └── REPORT_final_GROUP8.pdf                # report data strorytelling and technical analysis (static)
├── data/
│   └── rolling_sale_data.*       # NYC dataset (Nov 2024 – Oct 2025)
├── requirements.txt              # Optional: Python deps (for main.ipynb)
└── README.md
```
---

## Tech Stack & System Requirements
Python (recommended ≥ 3.9):

pandas, numpy

scikit-learn (Decision Tree & metrics)

matplotlib, seaborn

jupyter / jupyterlab

Power BI: Power BI Desktop (Windows) to open .pbix.

---

## Setup
Clone
```bash
git clone https://github.com/Gitzav/role_of_datapreparation.git
cd role_of_datapreparation
```
(Recommended) Create a virtual environment

```bash
python -m venv venv
```
### Windows
```bash
venv\Scripts\activate
```
### macOS / Linux
```bash
source venv/bin/activate
```
### Install dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run for visual.py
This project provides a complete exploratory data analysis (EDA) pipeline for NYC rolling real estate sales. The main entry point is the `run_analysis()` function inside `visual.py`, which performs:

* Data loading & cleaning
* Categorical & numerical distribution visualization
* Correlation heatmaps
* YEAR BUILT distribution
* PPSF–based noise/outlier detection
* Outlier summary
* Tax class analysis
* Log-transformed sale price analysis

Follow the steps below to run the analysis.
### **1. Install Dependencies**

Make sure you have Python 3.7+ installed.

Install required packages. All required packages are provided in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### **2. Run from Terminal**

You can directly execute the script:

```bash
python visual.py
```

### **3. Run Inside a Jupyter Notebook**

Import the function and call it:

```python
from visual import run_analysis

run_analysis('data/rolling_sale_data.xlsx')
```

### **4. Output**
Running the script will generate:

* Multiple plots (histograms, KDEs, bar charts)
* Correlation heatmap (with highlighted SALE PRICE row)
* PPSF outlier detection scatterplot
* Outlier summary bar chart
* Log-transformed price visualization

The output is displayed directly in your terminal and as visual charts.


---

## Author
Project: role_of_datapreparation

Topic: The Role of Data Preparation in Machine Learning

Contact: Group 8 · DSEB 65B · National Economics University

