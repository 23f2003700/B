# Machine Learning Basic Terminologies - Quick Reference

## **🤖 Core Machine Learning Terms**

### **Algorithm vs Model**
- **Algorithm:** The mathematical procedure/recipe (like a cooking recipe)
- **Model:** The trained result after applying algorithm on data (like the cooked dish)
- **Example:** Random Forest is algorithm, trained RandomForest on your data is model

### **Dataset Components**
- **Features (X):** Input variables/columns (age, salary, city)
- **Target/Label (y):** Output variable we want to predict (price, class)
- **Samples/Instances:** Individual rows of data
- **Dimensionality:** Number of features (columns)

### **Types of Data**
- **Numerical:** Numbers (age: 25, salary: 50000)
- **Categorical:** Categories (color: red/blue, gender: M/F)
- **Ordinal:** Ordered categories (rating: 1,2,3,4,5)
- **Nominal:** Unordered categories (city: Delhi/Mumbai/Chennai)

---

## **📊 Training and Testing Concepts**

### **Data Splitting**
```
Total Dataset (100%)
├── Training Set (80%) - Model learns from this
├── Validation Set (10%) - Tune hyperparameters
└── Test Set (10%) - Final evaluation (unseen data)
```

### **Why Split Data?**
- **Training:** Model learns patterns
- **Testing:** Check if model works on new, unseen data
- **Validation:** Choose best hyperparameters without touching test set

### **Overfitting vs Underfitting**
- **Overfitting:** Model memorizes training data, fails on new data
  - High accuracy on train, low on test
  - Solution: Regularization, more data, simpler model
- **Underfitting:** Model too simple, can't learn patterns
  - Low accuracy on both train and test
  - Solution: Complex model, more features, less regularization

### **Generalization**
- Model's ability to perform well on unseen data
- Goal of ML: Good generalization, not just training accuracy

---

## **🎯 Learning Types**

### **Supervised Learning**
- **Definition:** Learn from labeled examples (input-output pairs)
- **Types:**
  - **Classification:** Predict categories (spam/not spam)
  - **Regression:** Predict continuous values (house price)
- **Examples:** Email classification, sales forecasting

### **Unsupervised Learning**
- **Definition:** Find patterns in data without labels
- **Types:**
  - **Clustering:** Group similar data points
  - **Association:** Find relationships (people who buy X also buy Y)
  - **Dimensionality Reduction:** Reduce features while keeping info
- **Examples:** Customer segmentation, market basket analysis

### **Reinforcement Learning**
- **Definition:** Learn through trial and error with rewards/penalties
- **Examples:** Game playing (Chess, Go), Robot navigation

---

## **⚙️ Model Training Process**

### **Training Steps**
1. **Data Collection:** Gather relevant data
2. **Data Preprocessing:** Clean, transform, prepare data
3. **Feature Selection:** Choose important features
4. **Model Selection:** Choose appropriate algorithm
5. **Training:** Algorithm learns from training data
6. **Validation:** Test on validation set, tune parameters
7. **Testing:** Final evaluation on test set
8. **Deployment:** Use model in real world

### **Hyperparameters vs Parameters**
- **Parameters:** Model learns these during training (weights in neural network)
- **Hyperparameters:** We set these before training (learning rate, number of trees)

### **Cross-Validation**
- **K-Fold:** Split training data into K parts, train K times
- **Purpose:** More reliable performance estimate
- **Example:** 5-fold CV means 5 different train-validation splits

---

## **📈 Evaluation Metrics**

### **Classification Metrics**
- **Accuracy:** (Correct Predictions) / (Total Predictions)
- **Precision:** (True Positives) / (True Positives + False Positives)
- **Recall:** (True Positives) / (True Positives + False Negatives)
- **F1-Score:** Harmonic mean of Precision and Recall

### **Regression Metrics**
- **MSE (Mean Squared Error):** Average of squared differences
- **RMSE (Root MSE):** Square root of MSE (same units as target)
- **MAE (Mean Absolute Error):** Average of absolute differences
- **R² Score:** Proportion of variance explained (0 to 1, higher better)

### **Confusion Matrix**
```
              Predicted
           Yes    No
Actual Yes  TP   FN
       No   FP   TN
```
- **TP:** True Positive, **TN:** True Negative
- **FP:** False Positive, **FN:** False Negative

---

## **🔧 Data Preprocessing Terms**

### **Missing Values**
- **Types:** MCAR (Missing Completely at Random), MAR, MNAR
- **Handling:**
  - **Deletion:** Remove rows/columns with missing values
  - **Imputation:** Fill with mean/median/mode/predicted values

### **Feature Scaling**
- **Standardization:** (value - mean) / std_dev (Normal distribution)
- **Normalization:** (value - min) / (max - min) (0 to 1 range)
- **Why needed:** Some algorithms sensitive to feature scales

### **Encoding Categorical Variables**
- **Label Encoding:** Convert categories to numbers (Red=0, Blue=1, Green=2)
- **One-Hot Encoding:** Create binary columns for each category
- **Target Encoding:** Replace category with target mean

### **Feature Engineering**
- **Definition:** Creating new features from existing ones
- **Examples:** Age groups from age, day of week from date
- **Purpose:** Help model learn better patterns

---

## **🧠 Common Algorithms (Simple Explanations)**

### **Linear Regression**
- **What:** Draws best straight line through data points
- **Use:** Predict continuous values
- **Example:** Predict house price based on size

### **Logistic Regression**
- **What:** Uses S-shaped curve for classification
- **Use:** Binary classification (Yes/No, Spam/Not Spam)
- **Output:** Probability between 0 and 1

### **Decision Tree**
- **What:** Series of Yes/No questions leading to decision
- **Advantage:** Easy to understand and interpret
- **Disadvantage:** Can overfit easily

### **Random Forest**
- **What:** Combines many decision trees
- **Advantage:** Reduces overfitting, more accurate
- **How:** Each tree votes, majority wins

### **K-Means Clustering**
- **What:** Groups data into K clusters
- **Process:** Start with K centers, assign points, move centers, repeat
- **Use:** Customer segmentation, image compression

### **Support Vector Machine (SVM)**
- **What:** Finds best boundary to separate classes
- **Advantage:** Works well with high-dimensional data
- **Kernel Trick:** Can handle non-linear patterns

### **Neural Networks**
- **What:** Mimics brain neurons with connected layers
- **Components:** Input layer, hidden layers, output layer
- **Use:** Complex patterns, image recognition, NLP

---

## **📊 Important Concepts**

### **Bias-Variance Tradeoff**
- **Bias:** Error due to oversimplifying the model
- **Variance:** Error due to sensitivity to small changes in training data
- **Goal:** Balance both for best performance

### **Curse of Dimensionality**
- **Problem:** Too many features make data sparse
- **Effect:** Models perform poorly with too many dimensions
- **Solution:** Feature selection, dimensionality reduction

### **Feature Selection Methods**
- **Filter:** Use statistical tests (correlation, chi-square)
- **Wrapper:** Try different feature combinations
- **Embedded:** Algorithm selects features during training

### **Ensemble Methods**
- **Bagging:** Train multiple models on different data subsets (Random Forest)
- **Boosting:** Train models sequentially, each corrects previous errors (XGBoost)
- **Voting:** Combine predictions from multiple models

### **Regularization**
- **Purpose:** Prevent overfitting by penalizing complex models
- **L1 (Lasso):** Can set some features to zero (feature selection)
- **L2 (Ridge):** Shrinks all feature weights towards zero
- **Elastic Net:** Combines L1 and L2

---

## **🎲 Sampling and Validation**

### **Sampling Techniques**
- **Random Sampling:** Every sample has equal chance
- **Stratified Sampling:** Maintain proportion of different groups
- **Systematic Sampling:** Every nth sample
- **Cluster Sampling:** Sample entire groups/clusters

### **Validation Strategies**
- **Holdout:** Single train-validation-test split
- **K-Fold CV:** K different train-validation splits
- **Leave-One-Out:** Each sample used as validation once
- **Time Series Split:** For time-dependent data

---

## **🏗️ Model Deployment Terms**

### **Pipeline**
- **Definition:** Chain of data processing steps ending with model
- **Benefits:** Prevents data leakage, reproducible, production-ready
- **Example:** Scaling → Feature Selection → Model

### **Model Drift**
- **Data Drift:** Input data changes over time
- **Concept Drift:** Relationship between input-output changes
- **Solution:** Regular model retraining, monitoring

### **A/B Testing**
- **Purpose:** Compare model performance in real world
- **Method:** Split users into groups, use different models
- **Measure:** Business metrics (conversion, revenue)

---

## **🔍 Quick Formula Reference**

### **Basic Statistics**
- **Mean:** Σx / n
- **Variance:** Σ(x - mean)² / n
- **Standard Deviation:** √variance

### **Distance Metrics**
- **Euclidean:** √Σ(x₁ - x₂)²
- **Manhattan:** Σ|x₁ - x₂|
- **Cosine:** (A·B) / (|A||B|)

### **Information Theory**
- **Entropy:** -Σ(p × log₂(p))
- **Information Gain:** Entropy(parent) - Weighted_Entropy(children)

---

## **💡 Quick Tips for Viva**

### **Common Questions & Short Answers**
- **Q: What is machine learning?** A: Computer learning patterns from data to make predictions
- **Q: Difference between AI, ML, DL?** A: AI > ML > Deep Learning (subset relationship)
- **Q: Why split data?** A: To test model on unseen data and avoid overfitting
- **Q: What is cross-validation?** A: Multiple train-test splits for reliable performance estimate
- **Q: How to handle missing values?** A: Delete or impute with mean/median/mode
- **Q: What is regularization?** A: Technique to prevent overfitting by penalizing complexity

### **Key Points to Remember**
1. **Always explain in simple terms first, then technical details**
2. **Give real-world examples** (email spam, house prices, customer segmentation)
3. **Mention both advantages and disadvantages** of algorithms
4. **Connect concepts** (overfitting → regularization → cross-validation)
5. **Be confident about your ensemble approach** in the notebook

---

## **🚀 Last Minute Checklist**

- ✅ Understand supervised vs unsupervised vs reinforcement
- ✅ Know why we split data into train/validation/test
- ✅ Explain overfitting vs underfitting with examples
- ✅ Differentiate between parameters and hyperparameters
- ✅ Know common algorithms in 1-2 sentences each
- ✅ Understand evaluation metrics for classification vs regression
- ✅ Explain ensemble methods benefits
- ✅ Know preprocessing steps (scaling, encoding, missing values)
- ✅ Understand cross-validation purpose
- ✅ Be ready to give real-world examples for each concept

**Remember: Keep explanations simple, then add technical details if asked! 💪**
