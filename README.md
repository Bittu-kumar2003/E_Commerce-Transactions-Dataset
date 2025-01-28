# E_Commerce Transactions Dataset

## Overview
This project involves performing customer segmentation using clustering techniques. The tasks are split into three parts:

1. **Task 1: Data Cleaning and Preprocessing**  
   Prepare the data by cleaning and preprocessing to ensure accurate analysis.

2. **Task 2: Exploratory Data Analysis (EDA)**  
   Perform detailed exploratory analysis to understand patterns and trends in the dataset.

3. **Task 3: Customer Segmentation / Clustering**  
   Perform customer segmentation using clustering algorithms and evaluate the results.

---

## Task Details

### Task 1: Data Cleaning and Preprocessing
**Objective:**
- Clean and preprocess the `Customers.csv` and `Transactions.csv` datasets to remove inconsistencies and handle missing data.

**Steps:**
1. Check for missing values and impute or remove them appropriately.
2. Remove duplicate entries.
3. Normalize or scale numerical columns as required.
4. Merge datasets where applicable for further analysis.

**Outputs:**
- Cleaned datasets ready for analysis.
- Log of any data imputation or transformations performed.

---

### Task 2: Exploratory Data Analysis (EDA)
**Objective:**
- Understand the dataset through visualizations and descriptive statistics.

**Steps:**
1. Analyze customer demographics (age, location, etc.).
2. Explore transaction behaviors (frequency, quantity, total spend, etc.).
3. Visualize key trends using plots (e.g., histograms, scatter plots, bar charts).

**Outputs:**
- Insights into customer behavior and purchasing trends.
- Visualizations showcasing patterns in the data.
- Summary report of key findings.

---

### Task 3: Customer Segmentation / Clustering
**Objective:**
- Perform clustering on the customers using profile and transaction data.

**Steps:**
1. Merge `Customers.csv` and `Transactions.csv` to create a unified dataset.
2. Perform feature engineering (e.g., total spend, average transaction value).
3. Select a clustering algorithm (e.g., K-Means, DBSCAN, or Agglomerative Clustering).
4. Determine the optimal number of clusters using metrics like the Elbow Method or Silhouette Score.
5. Calculate the Davies-Bouldin Index (DB Index) to evaluate clustering performance.
6. Visualize clusters using relevant plots.

**Outputs:**
- Number of clusters formed.
- DB Index value and other evaluation metrics.
- Cluster visualizations.
- Final report on clustering results.

---

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-link>
   ```

---

## Usage
1. Place the datasets (`Customers.csv` and `Transactions.csv`) in the `data/` folder.
2. Run the Jupyter Notebook or Python scripts for each task sequentially.

   Example for Task 1:
   ```bash
   python task1_data_cleaning.py
   ```

3. View outputs in the `outputs/` folder.

---

## Evaluation Metrics
- **Davies-Bouldin Index (DB Index):** Measures the quality of clustering; lower values indicate better-defined clusters.
- **Silhouette Score:** Evaluates how similar an object is to its own cluster compared to other clusters.

---

## Results
- Task 1: Cleaned and preprocessed datasets.
- Task 2: Insights and visualizations of customer behaviors.
- Task 3: Clustering results with evaluation metrics and visualizations.

---

## Project Structure
```
├── data/
│   ├── Customers.csv
│   ├── Transactions.csv
│   ├── Products.csv
├── notebooks/
│   ├── task1_data_cleaning.ipynb
│   ├── task2_eda.ipynb
│   ├── task3_clustering.ipynb
├── outputs/
│   ├── cleaned_data.csv
│   ├── eda_visualizations/
│   ├── clustering_results/
├── requirements.txt
├── README.md
```

