# 🚀 NEO Sprint 1 — Near Earth Object (NEO) Data Analysis

A complete Exploratory Data Analysis (EDA) and preprocessing project on NASA Near Earth Object (NEO) dataset using Python.
This project focuses on understanding asteroid characteristics, detecting hazardous asteroids, handling outliers, scaling features, and preparing the dataset for Machine Learning models.

---

# 📌 Project Objective

The goal of this project is to:

* Analyze Near Earth Objects (NEOs)
* Understand asteroid behavior using EDA
* Detect patterns in hazardous asteroids
* Perform data preprocessing
* Prepare data for Machine Learning classification models

---

# 📂 Dataset Features

The dataset contains information about asteroids such as:

| Feature              | Description                          |
| -------------------- | ------------------------------------ |
| `est_diameter_min`   | Minimum estimated diameter           |
| `est_diameter_max`   | Maximum estimated diameter           |
| `relative_velocity`  | Velocity of asteroid                 |
| `miss_distance`      | Distance from Earth                  |
| `absolute_magnitude` | Brightness of asteroid               |
| `hazardous`          | Whether asteroid is hazardous or not |

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

# 📊 Workflow of the Project

## 1️⃣ Data Loading

* Imported dataset using Pandas
* Checked structure of dataset

---

## 2️⃣ Data Inspection

Performed:

* `df.info()`
* `df.describe()`
* missing value analysis
* duplicate value analysis
* datatype checking

---

## 3️⃣ Data Cleaning

* Removed unnecessary columns
* Converted target column into numeric format
* Created cleaned dataset

Dropped columns:

* `id`
* `name`
* `orbiting_body`
* `sentry_object`

---

# 📈 Exploratory Data Analysis (EDA)

## Univariate Analysis

Studied distributions of:

* asteroid diameter
* velocity
* miss distance
* magnitude

### Key Findings

✅ Most asteroids are small in size
✅ Diameter distributions are highly right-skewed
✅ Extreme outliers exist
✅ Hazardous asteroids are fewer in number

---

## Bivariate Analysis

Compared features with target variable (`hazardous`).

### Key Findings

✅ Hazardous asteroids tend to have larger diameters
✅ Smaller miss distances may indicate higher hazard probability
✅ Velocity contributes partially to hazard prediction

---

## Multivariate Analysis

### Correlation Heatmap

Analyzed feature relationships.

### Key Findings

✅ `est_diameter_min` and `est_diameter_max` are highly correlated
✅ Some features have weak correlation with target variable
✅ No severe multicollinearity issues found

---

# 📦 Outlier Detection

Used:

* Boxplots
* IQR Method

Formula used:

IQR = Q3 - Q1

Outlier limits:

Lower\ Limit = Q1 - 1.5(IQR)

Upper\ Limit = Q3 + 1.5(IQR)

### Observations

✅ Diameter and velocity contain many outliers
✅ Outliers can affect ML model performance

---

# ⚙️ Feature Scaling

Used:

```python
StandardScaler()
```

Scaling Formula:

genui{"math_block_widget_always_prefetch_v2":{"content":"z = \frac{x-\mu}{\sigma}"}}

Why scaling?

* Important for distance-based algorithms
* Prevents large-value features from dominating

Algorithms requiring scaling:

* KNN
* SVM
* Logistic Regression

---

# 🔀 Train-Test Split

Used:

```python
train_test_split()
```

Purpose:

* Train model on training data
* Evaluate on unseen test data

Used:

* `test_size=0.2`
* `random_state=42`
* `stratify=y`

---

# 🧠 Machine Learning Concepts Covered

This project demonstrates:

* Data preprocessing
* Exploratory Data Analysis
* Feature engineering
* Outlier handling
* Correlation analysis
* Scaling
* Train-test splitting
* Class imbalance understanding

---

# 📌 Major Insights from the Dataset

✅ Dataset is imbalanced
✅ Hazardous asteroids are fewer
✅ Diameter is an important feature
✅ Significant skewness exists in data
✅ Multiple outliers are present
✅ Hazard prediction is not perfectly separable

---

# 📷 Visualizations Included

* Histograms
* Bar charts
* Boxplots
* Correlation heatmap
* Pairplots

---

# 🚀 Future Improvements

Possible next steps:

* Apply Machine Learning models
* Perform feature selection
* Handle class imbalance using SMOTE
* Hyperparameter tuning
* Model evaluation

Suggested algorithms:

* Logistic Regression
* KNN
* Decision Tree
* Random Forest
* SVM
* XGBoost

---

# 📚 Learning Outcomes

After completing this project, you will understand:

* Complete EDA workflow
* Data preprocessing pipeline
* Visualization interpretation
* Outlier detection
* Feature scaling
* Correlation analysis
* ML-ready dataset preparation

---

# 👨‍💻 Author

Created as part of Machine Learning learning journey and Sprint Project.

---

# ⭐ If You Like This Project

Give it a ⭐ on GitHub and share your feedback!
