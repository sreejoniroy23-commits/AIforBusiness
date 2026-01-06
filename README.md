# AI for Business – Movie Revenue Prediction Project

## Project overview
This repository contains our final **AI for Business** project.  
The objective of the project is to **predict movie box-office performance** using a combination of:

- **Structured movie features** (e.g., metadata and numerical attributes)
- **Text-derived features**, including embeddings generated from movie reviews
- **Machine learning models** for prediction and evaluation
- **Unsupervised techniques** (clustering and BERTopic) for exploratory insights

The project is structured as a **three-notebook pipeline**, designed to be read and executed sequentially.

---

## Recommended reading & execution order

### 1 Data cleaning & exploratory analysis
**Notebook:** `Final_Notebook1_cleaning_and_EDA_Jan6.ipynb`

This notebook:
- Loads and cleans the raw/merged movie dataset
- Handles missing values and basic preprocessing
- Performs exploratory data analysis (EDA)
- Produces cleaned datasets used in later stages

 **Start here** to understand the data and preprocessing decisions.

---

### 2️ Supervised modelling (main analysis)
**Notebook:** `Notebook2_modelling_Jan6.ipynb`

This is the **core modelling notebook** .

It includes:
- Train/validation / test splits
- Feature preprocessing (imputation, scaling, PCA where applicable)
- Model training and evaluation (e.g., Random Forest, Neural Network)
- Use of saved preprocessing objects and trained models for reproducibility

---

### 3️ Clustering & BERTopic (unsupervised insights)
**Notebook:** `Clustering+BERTopic.ipynb`

This notebook provides:
- Unsupervised clustering of movies
- Topic modelling using BERTopic
- Exploratory insights into thematic patterns and segmentation

This notebook is **exploratory and complementary** to the supervised modelling results.

---

## External data & saved artifacts

All external files required to run the notebooks are located in the **root directory** of the repository.

### Core datasets
- `merged_cleaned_movies.parquet`  
  Cleaned and merged movie dataset.

- `X_features_with_embeddings.parquet`  
  Final feature matrix combining structured features with text-derived embeddings.

- `y_target.parquet`  
  Target variable used for supervised learning.

---

### Train/validation/test splits

**Feature sets**
- `X_train.parquet`
- `X_val.parquet`
- `X_test.parquet`

**Target sets**
- `y_train.parquet`
- `y_val.parquet`
- `y_test.parquet`

---

### Preprocessing variants

These files support the reproducibility of different modelling pipelines.

**Imputed feature sets**
- `X_train_imputed.parquet`
- `X_val_imputed.parquet`
- `X_test_imputed.parquet`

**Scaled feature sets**
- `X_train_scaled.parquet`
- `X_val_scaled.parquet`
- `X_test_scaled.parquet`

**PCA-transformed features**
- `X_val_pca.parquet`
- `X_test_pca.parquet`

---

### Saved preprocessing objects
Stored using `joblib` to avoid refitting during reruns.

- `imputor.joblib` – fitted imputer  
- `scaler.joblib` – fitted scaler  
- `pca.joblib` – fitted PCA transformer  

---

### Saved model
- `rf_model.joblib`  
  Trained Random Forest regression model used in the modelling notebook.

---

## How to run the project

### Option A — full pipeline 
1. Run `Final_Notebook1_cleaning_and_EDA_Jan4.ipynb`
2. Run `Notebook2_modelling_Jan4.ipynb`
3. Run `Clustering+BERTopic.ipynb`

### Option B — fastest path to results
- Open `Notebook2_modelling_Jan4.ipynb`
- Ensure all `.parquet` and `.joblib` files are present in the same directory
- Run all cells

---

## Environment & dependencies

The project is implemented in **Python (3.x)** and executed via **Jupyter notebooks**.

Key libraries used include:
- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `scipy`
- `joblib`

Modelling and NLP components additionally use:
- `tensorflow`, `keras_tuner`
- `torch`, `transformers`
- `shap`, `dice_ml`
- `bertopic`, `sentence_transformers`
- `umap`, `hdbscan`

---

## Notes for the grader
- All file paths assume the external files are stored **locally in the same directory** as the notebooks.
- Some modelling steps (e.g., hyperparameter tuning or embedding generation) are computationally intensive.  
  To reduce runtime, **preprocessed datasets and trained models are provided** and loaded where possible.

---

## Team
- Bustos, Virginia  
- Roy, Sreejoni  
- Ferrara, Ariana  
- Reusch, Benita
