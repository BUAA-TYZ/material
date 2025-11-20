# Quantitative Structure–Property Relationship (QSPR) Framework for Dynamic Polymer Design

This repository provides the **Quantitative Structure–Property Relationship (QSPR)** framework and associated computational code utilized in our research on **high-resolution DLP 3D printing of Hierarchical Dual-Dynamic Network (DRDCN) photoresins.**
The code implements a reproducible workflow for predicting material performance from molecular descriptors and accelerating the selection of optimal coordination crosslinkers.

---

## Role of Machine Learning & Problem Solved

The development of next-generation dynamic polymers requires exploring an enormous chemical space to identify ideal coordination nodes and network-forming components.
Traditional experimental screening is **time-consuming, resource-intensive, and limited in scalability.**

To address this challenge, we developed a **data-driven QSPR workflow** powered by gradient boosting regression (GBRT).
The machine learning model was specifically designed to:
- Predict key performance indicators of metal–ligand coordination systems
- Evaluate the balance between **stability, processability, and recyclability**
- Rapidly screen **hundreds of candidate coordination crosslinkers**
- Identify top-performing structures for experimental validation

This computational screening successfully highlighted **Terbium-Acetylacetonate (Tb-ACAC)** as the optimal coordination node—an ML-guided discovery that played a critical role in enabling the DRDCN system to achieve high mechanical performance and excellent 3D printability.

The QSPR framework presented here can be reused and extended for rational design and accelerated discovery of advanced dynamic materials.

---

## Note on Data Access

While the QSPR codebase is fully open-sourced for transparency, reproducibility, and future adaptation:

> The proprietary training dataset is not publicly released.

This dataset contains:
- Hundreds of meticulously curated metal-ligand coordination systems
- Experimentally validated performance metrics
- High-cost experimental data representing significant intellectual property

Because this dataset is a core asset enabling predictive accuracy and was derived from extensive laboratory effort, it cannot be shared publicly.
However, the computational framework here is complete and can be applied to any appropriate user-provided dataset.

---

## Repository Structure

```
.
├── data/
│   └── material.xlsx                    # Hided
│
├── outputs/
│   ├── figures/                         # Auto-generated plots
│   │   ├── cumulative_importance.png
│   │   ├── feature_importance.png
│   │   ├── pred_vs_actual.png
│   │   ├── residual_distribution.png
│   │   └── top10_predicted_vs_actual.png
│   │
│   ├── predicted_with_all_scores.xlsx   # Full model predictions
│   └── tables/                          # Raw numerical data used for figures
│       ├── cumulative_importance_data.csv
│       ├── feature_importance_data.csv
│       ├── pred_vs_actual_data.csv
│       ├── residual_distribution_data.csv
│       └── top10_predicted_vs_actual_data.csv
│
├── src/
│   ├── config.py
│   ├── data_utils.py
│   ├── metrics_utils.py
│   ├── models.py
│   ├── pipeline.py
│   └── plot_utils.py
│
├── main.py                              # Full QSPR training pipeline
└── README.md
```

---

## Usage

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the full QSPR pipeline

This trains a Gradient Boosting Regression Tree model and generates all plots + output tables:

```bash
python main.py --model gbrt
```

Outputs are saved automatically in:

```
outputs/figures/
outputs/tables/
outputs/predicted_with_all_scores.xlsx
```