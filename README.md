# Customer Segmentation using Clustering Techniques

This project demonstrates the process of performing customer segmentation using clustering techniques. The segmentation leverages both customer profile information and transaction data to group customers into meaningful clusters.

## Project Overview

### Objective
The primary objective of this project is to:
1. Perform customer segmentation using clustering algorithms.
2. Evaluate clustering performance using metrics like the Davies-Bouldin Index.
3. Visualize the clusters and provide insights.

### Features
- **Data Merging**: Combines data from `Customers.csv`, `Transactions.csv`, and `Products.csv` to create a unified dataset.
- **Feature Engineering**: Aggregates transaction data and enriches it with customer demographics.
- **Clustering**: Implements the K-Means clustering algorithm to segment customers.
- **Evaluation**: Calculates clustering metrics such as the Davies-Bouldin Index and Silhouette Score.
- **Visualization**: Generates scatter plots to visualize customer segments.

---

## Folder Structure
```
project-folder/
├── Customers.csv          # Customer profile data
├── Transactions.csv       # Transactional data
├── Products.csv           # Product data
├── Customer_Segments.csv  # Output file with clustering results
├── clustering_script.py   # Main Python script
└── README.md              # Project documentation
```

---

## Dataset Requirements

### Customers.csv
| Column Name | Description           |
|-------------|-----------------------|
| CustomerID  | Unique ID of customer |
| Age         | Age of customer       |
| Region      | Region of customer    |

### Transactions.csv
| Column Name | Description           |
|-------------|-----------------------|
| CustomerID  | Unique ID of customer |
| ProductID   | Unique ID of product  |
| Quantity    | Quantity purchased    |
| TotalValue  | Total value of purchase |

### Products.csv
| Column Name | Description           |
|-------------|-----------------------|
| ProductID   | Unique ID of product  |

---

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.7+
- Required Python libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

### Install Dependencies
Run the following command to install all dependencies:
```bash
pip install -r requirements.txt
```
Create a `requirements.txt` file if needed:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

## Usage

### Step 1: Prepare Data
Place the `Customers.csv`, `Transactions.csv`, and `Products.csv` files in the same directory as the script.

### Step 2: Run the Script
Execute the script using:
```bash
python clustering_script.py
```

### Step 3: Output
1. **Clustering Results**: A `Customer_Segments.csv` file containing customer IDs, features, and their assigned cluster.
2. **Clustering Metrics**: Displays the Davies-Bouldin Index and Silhouette Score in the console.
3. **Visualizations**: A scatter plot showing customer clusters.

---

## Key Sections in Code

### 1. Importing Libraries
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
```

### 2. Data Validation
Ensures required columns are present in the input datasets:
```python
def check_missing_columns(df, required_columns, file_name):
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"{file_name} is missing required columns: {missing_columns}")
```

### 3. Clustering Logic
Implements K-Means clustering and evaluates the model:
```python
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
customer_features['Cluster'] = kmeans.fit_predict(scaled_features)

# Evaluation
db_index = davies_bouldin_score(scaled_features, customer_features['Cluster'])
silhouette_avg = silhouette_score(scaled_features, customer_features['Cluster'])
```

### 4. Visualization
Generates a scatter plot of customer clusters:
```python
sns.scatterplot(
    x=scaled_features[:, 0],
    y=scaled_features[:, 1],
    hue=customer_features['Cluster'],
    palette='viridis',
    legend='full'
)
```

---

## Outputs
1. **Customer_Segments.csv**:
   Contains the following columns:
   - `CustomerID`
   - Aggregated features (e.g., `Quantity`, `TotalValue`)
   - Customer demographics (e.g., `Age`, `Region`)
   - Cluster assignment

2. **Clustering Metrics**:
   - Davies-Bouldin Index
   - Silhouette Score

3. **Cluster Visualization**:
   - Scatter plot of customer clusters.

---

## Evaluation Metrics
1. **Davies-Bouldin Index**:
   - A lower value indicates better clustering.
2. **Silhouette Score**:
   - Measures how similar an object is to its cluster compared to other clusters.

---

## Contribution
Feel free to fork the repository, create a branch, and submit a pull request. For major changes, open an issue first to discuss your ideas.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

