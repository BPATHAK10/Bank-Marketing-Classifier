# Bank-Marketing-Classifier

This project aims to build a machine learning classifier to predict whether a client will subscribe to a term deposit based on the Bank Marketing dataset. This is a common but challenging classification problem due to the significant class imbalance in the dataset.

## Features

### Data Preprocessing
The project includes a comprehensive data preprocessing pipeline to handle missing values, encode categorical variables, and perform feature engineering.

### Multicollinearity Handling
The code addresses multicollinearity among features by analyzing the correlation matrix and selectively dropping redundant variables.

### Dimensionality Reduction
Principal Component Analysis (PCA) is used to reduce the dimensionality of the dataset while retaining a high percentage of the original variance.

### Class Imbalance
The Synthetic Minority Oversampling Technique (SMOTE) is integrated into the preprocessing pipeline to address the class imbalance issue.

### Model Training
The project evaluates and tunes three different machine learning models:

- **Support Vector Machine (SVM)**
- **Logistic Regression**
- **Random Forest**

### Hyperparameter Tuning
`RandomizedSearchCV` with `StratifiedKFold` cross-validation is used to find the optimal hyperparameters for each model.

## Technologies

- **Python**: The core programming language for the project
- **Scikit-learn**: A robust library for machine learning tasks, including preprocessing, model selection, and evaluation
- **Pandas**: Used for data manipulation and analysis
- **Imbalanced-learn (imblearn)**: A library used to handle the class imbalance with SMOTE
- **Matplotlib/Seaborn**: Used for data visualization, such as generating the correlation matrix
