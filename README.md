#  Australian Weather Prediction — Big Data Analysis

> A big data project analyzing Australian weather patterns and predicting next-day rainfall using PySpark, scikit-learn, UMAP, and PCA.  
> Submitted as a university assignment at the **Ionian University**, Department of Informatics, January 2025.

---

##  Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Pipeline](#pipeline)
- [Machine Learning Results](#machine-learning-results)
- [Visualizations](#visualizations)
- [Authors](#authors)

---

## Overview

This project performs end-to-end analysis of the **weatherAUS** dataset to predict whether it will rain the next day in Australia. The workflow includes:

- Data loading and summary statistics
- Missing value imputation (mean for numeric, mode for categorical)
- Outlier detection and capping using Z-score
- Exploratory Data Analysis (EDA) with rich visualizations
- Dimensionality reduction via **PCA** and **UMAP** with K-Means clustering
- Binary classification using **PySpark MLlib**: Logistic Regression, Random Forest, and Gradient Boosted Trees (GBT)

---

## Dataset

**Source:** [**Rain in Australia Dataset**](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) -> weatherAUS.csv <br>
**Size:** ~145,460 rows × 23 columns  
**Target variable:** `RainTomorrow` (Yes / No)

Key features include temperature (min/max), rainfall, evaporation, sunshine hours, wind speed/direction, humidity, atmospheric pressure, and cloud coverage — recorded at both 9am and 3pm.

---

## Project Structure

```
.
├── weather_prediction.ipynb        # Main Jupyter notebook (full pipeline)
├── weatherAUS.csv                  # Raw dataset
├── requirements.txt                # Python dependencies
├── docker-compose.yml              # Docker setup for PySpark (Jupyter all-spark-notebook)
├── BigData-Report.pdf              # Full academic report (Greek)
├── weather_prediction_notebook.pdf
└── README.md
```

---

## Setup & Installation

### Option 1 — Docker (Recommended for PySpark)

Requires [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/).

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Place weatherAUS.csv inside a /src folder
mkdir src
cp weatherAUS.csv src/

# 3. Start the Jupyter/Spark environment
docker-compose up
```

Then open the URL shown in the terminal (e.g. `http://127.0.0.1:8888/...`) and run `weather_prediction.ipynb`.

### Option 2 — Local Python Environment

Requires Python 3.9+ and Java 8+ (needed for PySpark).

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook weather_prediction.ipynb
```

---

## Pipeline

```
Rain in Australia Dataset -> weatherAUS.csv
     │
     ▼
1. Data Loading & Summary Statistics
     │
     ▼
2. Missing Value Imputation
   ├── Numeric columns  → filled with column mean
   └── Categorical cols → filled with column mode
     │
     ▼
3. Outlier Handling (Z-score capping, threshold = 3)
     │
     ▼
4. Exploratory Data Analysis
   ├── Histograms, Boxplots, Correlation Heatmap
   ├── Rainfall over Time
   └── RainToday / RainTomorrow distribution
     │
     ▼
5. Dimensionality Reduction + Clustering
   ├── PCA (2 components, ~41.5% cumulative variance)
   └── UMAP (n_neighbors=15, min_dist=0.1) + K-Means (k=5)
     │
     ▼
6. Machine Learning (PySpark MLlib)
   ├── Logistic Regression
   ├── Random Forest
   └── Gradient Boosted Trees (GBT)
```

---

## Machine Learning Results

All models were trained on an 80/20 train-test split. Performance is measured using **Area Under the ROC Curve (AUC)**:

| Model                   | AUC    |
|-------------------------|--------|
| Logistic Regression     | **0.851** ✅ |
| Gradient Boosted Trees  | 0.703  |
| Random Forest           | 0.661  |

Logistic Regression achieved the best performance, with strong separation between rainy and non-rainy days. Categorical features (location, wind direction) were encoded using `StringIndexer`, and all features were assembled into a single vector via `VectorAssembler`.

---

## Visualizations

The notebook produces the following plots:

| Visualization | Description |
|---|---|
| Boxplots | Distribution and outliers for all numeric features |
| Histograms | Feature distributions after cleaning |
| Correlation Heatmap | Pairwise correlation of all numerical columns |
| Rainfall Over Time | Average daily rainfall from 2008–2017 |
| RainToday / RainTomorrow Distribution | Class balance of the target variable |
| UMAP + K-Means | Non-linear 2D embedding with 5 clusters |
| PCA + K-Means | Linear 2D embedding with 5 clusters |

---

## Dependencies

See [`requirements.txt`](requirements.txt):

```
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
umap-learn
umap
```

PySpark (v3.5.0) is provided via the Docker image `jupyter/all-spark-notebook`.

---

## Authors

| Name | 
|---|
| Konstantinos Kafteranis |
| Stergios Moutzikos |
| Christos Kostakis |

*Ionian University — Big Data Management, January 2025*
