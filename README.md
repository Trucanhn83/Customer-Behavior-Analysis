# E-commerce Customer Behavior Analysis

Exploratory data analysis (EDA) and churn prediction on an e-commerce customer-behavior dataset — from data cleaning, outlier detection, and statistical analysis to visualizing churn drivers and building classification models.

> Project for **DAB303 – Marketing Analytics** (Project 3).

---

## 1. Overview

The goal is to understand customer behavior in an e-commerce setting and identify the factors that drive **customer churn**. The notebook walks through a structured EDA framework, analyzes the dependent variable (`churn`) against independent variables, and trains a classifier to predict churn and rank feature importance.

## 2. Data

- **Source:** `Customer_Behavior.csv` (downloaded from Kaggle)
- **Size:** 3,333 rows × 20 columns
- **Target (`churn`):** binary — `0` = retained, `1` = churned
- **Class balance:** imbalanced (~83.5% non-churn, ~16.5% churn)

**Feature groups:**
- *Account / engagement:* `account_length`, `desktop_sessions`, `app_sessions`, `session_duration`, `add_to_wishlist`, `promotion_clicks`, `sale_product_views`, `total_product_detail_views`, `product_detail_view_per_app_session`, `add_to_cart_per_session`
- *Transactions / value:* `desktop_transactions`, `app_transactions`, `avg_order_value`, `discount_rate_per_visited_products`
- *Service / categorical:* `customer_service_calls`, `location_code`, `credit_card_info_save`, `push_status`
- *ID:* `user_id`

> ⚠️ `Customer_Behavior.csv` is **not included** in this repo. Download the dataset and update the file path in the *How to Run* section.

## 3. Project Structure

```
.
├── P_3__Customer_Behavior_Analysis__PROBLEM_.ipynb   # Main notebook (code + analysis)
├── Customer_Behavior_Analysis_.pptx                   # Results presentation (13 slides)
├── Customer_Behavior.csv                              # Dataset (add it yourself)
└── README.md
```

## 4. Workflow

**a. Load & Clean**
Load the CSV; standardize column names (replace spaces with underscores); convert comma-decimal string columns (e.g. `avg_order_value`, `discount_rate_per_visited_products`) to `float`.

**b. Statistical Analysis**
Descriptive statistics (`.describe()`) for numeric and non-numeric columns.

**c. Data Quality Checks**
Check missing values (with a `seaborn` heatmap) and verify `user_id` uniqueness / duplicate records.

**d. Outlier Detection**
Define an IQR-based `detect_outliers()` function (Q1 − 1.5·IQR, Q3 + 1.5·IQR); optionally remove outliers; visualize each numeric variable with `plotly` histograms and box plots, plus a combined box-plot grid.

**e. Churn vs. Features Analysis**
- Group numeric variables by `churn` and compare distributions (box plots by churn).
- **Chi-square test** of `location_code` vs `churn` (result: p ≈ 0.42 → location is *not* a significant predictor).
- **One-hot encode** categorical variables (`credit_card_info_save`, `push_status`, `location_code`).
- Correlation of all features against `churn`.

**f. Modeling**
Train/test split, then classification with **RandomForestClassifier** (the slides also reference a DecisionTreeClassifier baseline); evaluate with `classification_report`; extract **feature importance**.

**g. Categorical Analysis**
Plot categorical variables and their relationship to churn.

## 5. Key Results & Insights

- **Model performance:** RandomForest reaches **~93% accuracy** (precision 0.94 / recall 0.99 on the majority class; precision 0.90 / recall 0.67 on the churn class — recall on churn is limited by class imbalance).
- **Most important features:** `desktop_sessions`, `customer_service_calls`, and `app_sessions` — engagement and support interactions strongly influence churn.
- **Credit card info saved → much lower churn.** Encouraging customers to save card details may reduce churn.
- **Push notifications enabled → lower churn.** Driving push opt-in may aid retention.
- **Location is not a strong predictor** (similar churn rates across location codes; confirmed by the chi-square test).
- Several behavioral features (promotion clicks, app transactions, add-to-wishlist) contain notable outliers worth considering in further modeling.

## 6. Requirements

- Python 3.x

```bash
pip install pandas numpy matplotlib seaborn plotly scipy scikit-learn
```

## 7. How to Run

1. Place `Customer_Behavior.csv` in the project folder.
2. Open `P_3__Customer_Behavior_Analysis__PROBLEM_.ipynb` (Jupyter Notebook / JupyterLab / VS Code).
3. **Update the data path** in the CSV-reading cell — replace the original absolute path:

   ```python
   cb_df = pd.read_csv('Customer_Behavior.csv')   # instead of the original D:\... path
   ```

4. Run the cells in order from top to bottom.

## 8. Tech Stack

`pandas` · `numpy` · `matplotlib` · `seaborn` · `plotly` · `scipy` (chi-square) · `scikit-learn` (RandomForest, train/test split, metrics)
