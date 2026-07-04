# End-to-End Customer Churn Prediction Pipeline using XGBoost

A machine learning classification pipeline built to predict customer churn. This repository demonstrates production-grade software engineering practices applied to data science, utilizing robust data preprocessing, Scikit-Learn pipelines, and the high-performance XGBoost framework.

## Key Features
* **Modular Codebase:** Clean separation of concerns with dedicated modules for preprocessing (`preprocessing.py`), model configuration (`model.py`), and operational orchestration (`main.py`).
* **Zero Data Leakage:** Implements robust Scikit-Learn `ColumnTransformer` pipelines ensuring data scaling and categorical encoding are strictly fitted onto the training fold.
* **Production Ready:** Preprocessing and model training are encapsulated into a single unified pipeline artifact, making it trivial to serialize (`joblib`/`pickle`) and serve via a REST API (e.g., FastAPI).
* **Robust Imputation:** Automatically handles missing numerical observations via median imputation and handles unseen categorical variables gracefully without throwing runtime faults.

---

## Repository Structure

```text
├── preprocessing.py   # Ingests raw data and constructs the ColumnTransformer pipeline
├── model.py           # Configures the XGBoost classifier and bundles it into the master Pipeline
├── main.py            # The central orchestrator that manages data splitting, training, and evaluation
├── requirements.txt   # Standard environment dependency specifications
└── README.md          # Comprehensive project documentation
