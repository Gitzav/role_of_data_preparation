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

---

## Tech Stack & System Requirements
Python (recommended ≥ 3.9):

pandas, numpy

scikit-learn (Decision Tree & metrics)

matplotlib, seaborn

jupyter / jupyterlab

Power BI:

Power BI Desktop (Windows) to open .pbix.
---

## Setup
Clone
git clone https://github.com/Gitzav/role_of_datapreparation.git
cd role_of_datapreparation
(Recommended) Create a virtual environment

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
Install dependencies

If you have requirements.txt:

pip install -r requirements.txt
Or install the essentials:

pip install pandas numpy scikit-learn matplotlib seaborn jupyter
Sample requirements.txt (optional):

pandas>=2.0
numpy>=1.23
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.13
jupyterlab>=4.0
---

## How to Run

---

## Author
Project: role_of_datapreparation

Topic: The Role of Data Preparation in Machine Learning

Contact: [Your Name] · [Email / GitHub / LinkedIn]

If you find this useful, please ⭐ the repo and feel free to open issues/PRs.
