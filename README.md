# MLP Viva Complete Guide - Tomorrow Evening Preparation

## **🎯 Actual Viva Questions & Answers**

### **1. What is machine learning?**
**Answer:** Machine Learning is a subset of Artificial Intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or classifications on new, unseen data.

**Types:** Supervised, Unsupervised, Reinforcement Learning

---

### **2. What are the types of machine learning?**
**Answer:**
- **Supervised Learning:** Uses labeled data (input-output pairs)
  - Examples: Classification, Regression
- **Unsupervised Learning:** Finds patterns in unlabeled data
  - Examples: Clustering, Dimensionality Reduction
- **Reinforcement Learning:** Learns through trial and error with rewards/penalties
  - Examples: Game playing, Robotics

---

### **3. Name any three unsupervised algorithms**
**Answer:**
1. **K-Means Clustering** - Groups data into k clusters
2. **Hierarchical Clustering** - Creates tree-like cluster structure
3. **DBSCAN** - Density-based clustering
4. **PCA (Principal Component Analysis)** - Dimensionality reduction
5. **Apriori Algorithm** - Association rule mining

---

### **4. How KMeans work?**
**Answer:**
**Steps:**
1. **Initialize:** Choose k random centroids
2. **Assign:** Each point goes to nearest centroid
3. **Update:** Move centroids to center of their clusters
4. **Repeat:** Steps 2-3 until convergence
5. **Stop:** When centroids don't move significantly

**Formula:** Minimize Within-Cluster Sum of Squares (WCSS)
```
WCSS = Σ(distance from point to its centroid)²
```

---

### **5. Coding Question: Load Digits + KMeans + Optimal K**
**Complete Code:**
```python
# Import libraries
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
digits = load_digits()
X = digits.data
print(f"Dataset shape: {X.shape}")  # (1797, 64)

# Find optimal K using Elbow Method
wcss = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Plot results
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(k_range, wcss, 'bo-')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'ro-')
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.show()

# Best k
best_k = k_range[np.argmax(silhouette_scores)]
print(f"Optimal k: {best_k}")
```

---

### **6. Explain the preprocessing in your notebook**
**Answer based on your notebook:**
```python
# Your preprocessing steps:
1. **Missing Value Imputation:**
   - Categorical: Mode (most frequent value)
   - Numerical: Median (robust to outliers)

2. **Label Encoding:**
   - Converted categorical text to numbers
   - Combined train+test for consistent encoding

3. **Standard Scaling:**
   - Normalized numerical features (mean=0, std=1)
   - Applied fit on train, transform on test

4. **Target Transformation:**
   - Square root transformation to reduce skewness
   - Handles zeros naturally, compresses extreme values
```

---

### **7. Does your dataset contain outliers? How did you remove them?**
**Answer:**
"Yes, my dataset had extreme outliers - purchase values ranged from $0 to $23+ billion. I didn't remove outliers but handled them through:
1. **Square root transformation** - compressed extreme values
2. **Tree-based models** - naturally robust to outliers
3. **Ensemble approach** - reduces impact of outliers"

---

### **8. If dataset has many outliers, which model is suitable?**
**Answer:**
**Robust Models:**
- **Tree-based:** Random Forest, XGBoost, Decision Trees
- **Ensemble methods:** Voting, Bagging
- **Robust algorithms:** Huber Regression, RANSAC
- **Non-parametric:** KNN, SVM with RBF kernel

**Avoid:** Linear Regression, Naive Bayes (sensitive to outliers)

---

### **9. Why did you choose Random Forest? Parameter selection?**
**Answer:**
"Actually, I imported RandomForest but used **XGBoost and LightGBM** instead because:
- Better performance on zero-inflated data
- Superior handling of mixed feature types
- Built-in regularization

**Parameter selection for XGBoost:**
- n_estimators: 3000-3400 (high capacity)
- max_depth: 13-15 (moderate complexity)
- learning_rate: 0.007-0.009 (conservative)
- Based on research papers and empirical testing"

---

### **10. What is the function of r2_score?**
**Answer:**
**R² Score (Coefficient of Determination):**
- **Range:** -∞ to 1
- **Formula:** R² = 1 - (SS_res / SS_tot)
- **Interpretation:**
  - 1.0 = Perfect prediction
  - 0.0 = Model same as mean baseline
  - <0 = Model worse than baseline

**Code:**
```python
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
print(f"R² Score: {r2:.3f}")
```

---

### **11. What are solutions for overfitting?**
**Answer:**
1. **Regularization:** L1 (Lasso), L2 (Ridge), Elastic Net
2. **Cross-validation:** K-fold validation
3. **Early stopping:** Stop training when validation error increases
4. **Dropout:** Random neuron deactivation (Neural Networks)
5. **Data augmentation:** Increase training data
6. **Feature selection:** Remove irrelevant features
7. **Ensemble methods:** Combine multiple models
8. **Pruning:** Simplify decision trees

---

### **12. How is root node selected in decision tree?**
**Answer:**
**Selection Criteria:**
1. **Information Gain:** Choose feature with highest info gain
2. **Gini Impurity:** Minimize Gini impurity
3. **Chi-square:** Statistical significance test

**Formula (Gini):**
```
Gini = 1 - Σ(probability of class i)²
```

**Process:** Try all features, calculate impurity reduction, choose best split

---

### **13. What is the use of pipeline?**
**Answer:**
**Benefits of Pipeline:**
1. **Prevents data leakage:** Proper train-test separation
2. **Code organization:** Single object for all preprocessing
3. **Reproducibility:** Consistent preprocessing steps
4. **Production ready:** Deploy entire pipeline as one unit

**Code:**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor())
])
pipeline.fit(X_train, y_train)
```

---

### **14. Pure node vs Impure node concept?**
**Answer:**
- **Pure Node:** All samples belong to same class (Gini = 0)
- **Impure Node:** Samples from multiple classes mixed

**Examples:**
- Pure: [Class A, Class A, Class A] → Gini = 0
- Impure: [Class A, Class B, Class A] → Gini > 0

**Goal:** Decision tree splits to create pure nodes

---

### **15. Load digits dataset size + SVM classification**
**Answer:**
```python
# Dataset size
from sklearn.datasets import load_digits
digits = load_digits()
print(f"Shape: {digits.data.shape}")  # (1797, 64)
print(f"Classes: {len(digits.target_names)}")  # 10 classes (0-9)

# SVM Classification
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

# Train SVM
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)

# Predictions
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
```

---

### **16. Different optimization functions in neural networks?**
**Answer:**
1. **SGD (Stochastic Gradient Descent):** Basic optimizer
2. **Adam:** Adaptive learning rate, momentum
3. **RMSprop:** Adaptive learning rate
4. **Adagrad:** Accumulates gradients
5. **AdaDelta:** Extension of Adagrad
6. **Nadam:** Adam + Nesterov momentum

**Most popular:** Adam (good default choice)

---

### **17. ReLU function input and output?**
**Answer:**
**ReLU (Rectified Linear Unit):**
- **Formula:** f(x) = max(0, x)
- **Input:** Any real number (-∞ to +∞)
- **Output:** Non-negative numbers (0 to +∞)

**Examples:**
- Input: -5 → Output: 0
- Input: 0 → Output: 0  
- Input: 3 → Output: 3

---

### **18. One vs One and One vs All approach?**
**Answer:**
**Multi-class classification strategies:**

**One vs All (OvA):**
- Creates n binary classifiers (n = number of classes)
- Each classifier: "Class A vs All Other Classes"
- Example: For 3 classes → 3 classifiers

**One vs One (OvO):**
- Creates n(n-1)/2 binary classifiers
- Each classifier: "Class A vs Class B"
- Example: For 3 classes → 3 classifiers

**Usage:** SVM uses OvO by default, Logistic Regression uses OvA

---

### **19. What is t-test?**
**Answer:**
**T-test:** Statistical test to compare means

**Types:**
1. **One-sample t-test:** Compare sample mean to population mean
2. **Two-sample t-test:** Compare means of two groups
3. **Paired t-test:** Compare before/after measurements

**Formula:**
```
t = (sample_mean - population_mean) / (standard_error)
```

**Usage:** Determine if differences are statistically significant

---

## **🚀 Additional Important Topics for Viva**

### **Your Notebook Specific Questions**

**Q: Why ensemble of 6 models?**
**A:** "Ensemble reduces overfitting and improves generalization. Different models capture different patterns - XGBoost focuses on gradient optimization, LightGBM on speed and memory efficiency. Weighted combination gives better results than single model."

**Q: Why square root transformation?**
**A:** "Target had extreme skewness (0 to $23B). Square root compresses large values, handles zeros naturally, and makes distribution more symmetric for ML algorithms."

**Q: Why Label Encoding instead of One-Hot?**
**A:** "Tree-based models handle ordinal encoding well without false ordinality assumptions. One-hot would create 200+ columns causing memory issues and curse of dimensionality."

### **Quick ML Concepts Review**

**Bias-Variance Tradeoff:**
- High Bias: Underfitting
- High Variance: Overfitting
- Goal: Balance both

**Cross-Validation:**
- K-Fold: Split data into k parts
- Leave-One-Out: Each sample is test set once
- Stratified: Maintains class distribution

**Evaluation Metrics:**
- **Classification:** Accuracy, Precision, Recall, F1-Score
- **Regression:** MSE, RMSE, MAE, R²

**Feature Selection:**
- **Filter:** Chi-square, correlation
- **Wrapper:** Forward/Backward selection
- **Embedded:** Lasso, Random Forest importance

### **Common Coding Questions**

**1. Train-Test Split:**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**2. Standard Scaling:**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**3. Model Training:**
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### **Final Tips for Tomorrow**

1. **Practice the load_digits + KMeans code** - High chance of being asked
2. **Know your notebook inside-out** - Explain every preprocessing step
3. **Be confident about ensemble approach** - Your strength
4. **Understand tree-based models deeply** - Your main algorithms
5. **Practice basic ML coding** - train_test_split, scaling, evaluation

### **Last Minute Revision Checklist**

- ✅ Machine Learning definition and types
- ✅ Supervised vs Unsupervised vs Reinforcement
- ✅ K-Means algorithm steps
- ✅ Your preprocessing pipeline explanation
- ✅ Overfitting solutions
- ✅ Decision tree concepts
- ✅ Pipeline benefits
- ✅ Ensemble methods advantages
- ✅ Evaluation metrics (R², accuracy)
- ✅ Basic sklearn syntax

**You got this! Your ensemble approach is sophisticated - own it with confidence! 🚀**
