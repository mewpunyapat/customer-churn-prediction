# customer-churn-prediction

## Dataset  
The dataset used in this project is publicly available on [Kaggle: Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

It contains information about a telecom company's customers and whether they have churned.
The dataset includes demographic, service usage, and account information, such as:
- Contract type, tenure, and monthly charges
- Internet services, phone plans, streaming usage
- Demographic attributes like gender, seniority, and partner status

The target variable is binary: churned or not churned, making this suitable for **Binary Classification** task

---

## Objectives  
The main objective of this project is:

> To develop a machine learning system that accurately predicts customer churn with high recall so that the business can proactively intervene and reduce revenue loss.

To achieve this, the project is broken down into the following sub-goals:

1. Perform in-depth **Exploratory Data Analysis (EDA)** to understand patterns and distributions across features.
2. Engineer new predictive features to enhance model performance.
3. Train multiple classification models and evaluate them with a recall-first mindset while ensuring the precision is acceptable
4. Use hyperparameter tuning to improve model performance
5. Visualize results with ROC, PR curves, and feature importance
6. Summarize actionable insights and determine which model is optimal for deployment

---

## Key Insights from EDA
1. Tenure has negative correlation with churn rate. Customers with short tenure (0-6 months) exhibit a higher churn rate compared to long-term customers, suggesting that new customers may still feel unfamiliar and evaluate services quality.
2. Month-to-month contract type makes up the majority of customer base. This type of contract also exhibits the highest churn rate compared to others due to its flexiblity and low commitment. It also shows that customer who has month-to-month contract also have low tenure
3. High monthly charges correlate with an increase in churn likelihood. The business can focus this segment by monitoring satisfaction or offer some incentives to make them feel worth of what they pay
4. Fiber optic internet users show a 2x higher churn rate than DSL users, suggesting service-related dissatisfaction.
5. Electronic check significantly increase churn rate
6. The presence of Technical support with other add on services is a great factor that reduce churn rate

---

## Engineered Features
From the customer data, the following features were extracted:
- Feature 1: Fiber_NoTechSupport 
- Feature 2: M2M_Electronic check
- Feature 3: TenureGroup (categorized tenure into 0-6, 6-12, 12+ months) – higher churn in 0-6 group
- Feature 4: FiberOptic_StreamingTv
  
This feature engineering boosted the ROC AUC by 15% and improved the F1 score uplift from the baseline (0.574) to 0.624, a 8.7% enhancement

---

## Model Selection
Models were evaluated using ROC AUC due to the binary classification task and imbalanced labels. Three models (Logistic Regression, Random Forest, XGBoost) were tuned with GridSearchCV over 5-fold stratified cross-validation. The best-performing model is Logistic Regression with the following parameters:

<pre> { "C": 0.1, "penalty": "l2", "solver": "lbfgs", "max_iter": 500, "class_weight": "balanced" } </pre>
