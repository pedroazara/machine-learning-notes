# Machine Learning Notes

> A structured study repo covering classical ML from theory to deployment — including a working PyQt5 desktop app.

---

## Topics

| # | Topic | Status |
|---|-------|--------|
| 1 | Supervised vs Unsupervised Learning · Train/Val/Test Split | ✅ |
| 2 | Linear Regression · Least Squares · Gradient Descent | ✅ |
| 3 | Logistic Regression · MLE · Decision Boundaries | ✅ |
| 4 | K-Nearest Neighbors (KNN) | ✅ |
| 5 | Bias–Variance Tradeoff | ✅ |
| 6 | Regularization · L1 & L2 Penalties | ✅ |
| 7 | Support Vector Machines · Max Margin · Kernel Trick | ✅ |
| 8 | Decision Trees · Random Forests | ✅ |
| 9 | Model Evaluation · Confusion Matrix · ROC/AUC · F1 | ✅ |
| 10 | Cross-Validation · Hyperparameter Tuning | ✅ |

---

## Repository Layout

```
machine-learning-notes/
│
├── notebooks/                  # Algorithm deep-dives
│   ├── knn.ipynb
│   ├── decision_tree.ipynb
│   ├── decision_trees_complete.ipynb
│   └── multiclass_logistic_regression.ipynb
│
├── theory/                     # Math-first notebooks
│   └── autodiff.ipynb          # Automatic differentiation from scratch
│
├── projects/                   # End-to-end applied projects
│   ├── sonar_rock_mine.ipynb   # Binary classification · sonar signals
│   └── drug_guide/             # Full desktop application (see below)
│
├── scripts/
│   └── pipelines.py            # Reusable sklearn Pipeline builders
│
├── data/                       # Datasets used across notebooks
│   ├── iris.csv
│   ├── cancer_tumor_data_features.csv
│   ├── sonar.all-data.csv
│   └── ObesityDataSet_raw_and_data_sinthetic.csv
│
└── syllabus.md
```

---

## Projects

### Drug Prescription Assistant

A desktop app built with PyQt5 that wraps a trained scikit-learn Decision Tree to suggest drug prescriptions based on patient data.

```
projects/drug_guide/
├── drug_guide.ipynb    # Data exploration, training, evaluation
├── drug_model.pkl      # Serialized model (joblib)
├── model_meta.yaml     # Model card: accuracy, dataset, warnings
└── app.py              # PyQt5 GUI — run with: python app.py
```

**Model stats**

| Attribute | Value |
|-----------|-------|
| Algorithm | Decision Tree (entropy, max_depth=4) |
| Dataset | IBM Drug200 — 200 patients, 5 drugs |
| Accuracy | **98.33%** |
| Key feature | Na/K ratio (strongest predictor) |

**App features**
- Dynamic form built at runtime from the model's feature list — load any compatible `.pkl` at runtime
- Decision path explainability — shows every split the tree took to reach its answer
- Confidence bar per prediction
- Model info dialog with full metadata and clinical warnings

**Run it**

```bash
pip install pyqt5 joblib scikit-learn pyyaml numpy
python projects/drug_guide/app.py
```

---

### Sonar Rock vs Mine

Binary classification on the UCI sonar dataset (208 samples, 60 frequency features).  
Covers feature scaling, logistic regression, evaluation with confusion matrix and ROC curve.

---

## Setup

```bash
# Clone
git clone <repo-url>
cd machine-learning-notes

# Create environment (optional but recommended)
python -m venv .venv && source .venv/bin/activate   # Linux/macOS
python -m venv .venv && .venv\Scripts\activate       # Windows

# Install dependencies
pip install -r ../requirements.txt   # core: numpy, pandas, scikit-learn, matplotlib, seaborn
```

Launch Jupyter and open any notebook:

```bash
jupyter notebook
```

---

## Stack

`Python` · `NumPy` · `Pandas` · `scikit-learn` · `Matplotlib` · `Seaborn` · `PyQt5`
