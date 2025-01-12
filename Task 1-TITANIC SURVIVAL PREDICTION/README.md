# TITANIC SURVIVAL PREDICTION:

# Task Description:

* Use the Titanic dataset to build a model that predicts whether a passenger on Titanic survived or not. This is a classic beginner project with readily available data.

* The dataset typically used for this project contains nformation about individual passengers such as their age, gender, ticket class, fare, cabin, and whether or not they survived.

# Overview:
- Developed a predictive model to determine survival likelihood on the Titanic dataset.
- Utilized several classification models to predict survival
- Implemented the project using Python with Pandas, NumPy, Scikit-learn, and Seaborn for analysis and model development.

# About the Dataset:

The Titanic Dataset [link](https://www.kaggle.com/datasets/brendan45774/test-file) is a dataset curated on the basis of the passengers on titanic, like their age, class, gender, etc to predict if they would have survived or not. It contains both numerical and string values. It has 12 predefined columns which are as below:
- Passenger ID - To identify unique passengers
- Survived - If they survived or not
- PClass - The class passengers travelled in
- Name - Passenger Name
- Sex - Gender of Passenger
- Age - Age of passenger
- SibSp - Number of siblings or spouse
- Parch - Parent or child
- Ticket - Ticket number
- Fare - Amount paid for the ticket
- Cabin - Cabin of residence
- Embarked - Point of embarkment

# Project Details:
- Objective: Predict survival probability of passengers aboard the Titanic.
- Model Used: 'LogisticRegression,'DecisionTree','RandomForest','Bagging','Adaboost', 'Gradient Boosting', 'XGBoost','Support Vector Machine','K-Nearest Neighbors',  'Naive Bayes Gaussian' and 'Naive Bayes Bernoullies' from Scikit-learn along with 'Voting Classifier' for the best prediction model.
- Best Model: Support Vector Machine Model
- Accuracy Achieved: 98%.

# Key Features:
- Conducted data preprocessing including handling missing values and feature engineering.
- Used Standarization technique to normalise the data before model training.
- Trained several classification models to predict survival, most of which performed well, likely due to the relatively small dataset size. Out of which, SVM model gave 98% accuracy and KNN model gave 96.4% accuracy
- Model evaluation involved accuracy measurement and potentially other relevant metrics.


  # Exploratory Data Analysis (EDA):

  # Finding the Missing values:


  # Handling the Missing values:

  # Distribution of Survival:

  # Distribution for survival distribution by Sex:


  # Distribution for Survival by Passenger Class:

  # Distribution for Survival by Port of Embarkation:

  # correlation matrix:


  # Model Training:


  # 10-fold Cross Validation Results:


  # Conclusion:
- Our analysis unveiled key insights into the Titanic dataset. We addressed missing values by filling null entries in the Age and Fare columns with medians due to the presence of outliers, while the Cabin column was discarded due to huge amount of null values.
- Notably, All the female passengers survived and all the male passengers not survived. 
- Furthermore, we observed that Passenger class 3 had the highest number of deaths and most of the Passenger class 1 have survived.
- Most of the Passengers from Queenstown had a higher survival rate compared to those from Southampton.
- In this Titanic Survival Prediction analysis, we have explored various aspects of the dataset to understand the factors influencing survival. 
- We found that only 152 passengers i.e. 36.4% of the passengers survived the crash, with significant differences in survival rates among different passenger classes, genders, and age groups. 
- The dataset also revealed that certain features, such as Fare and embarkation location, played a role in survival. 
- We trained several classification models to predict survival, most of which performed well, likely due to the relatively small dataset size. Out of which, SVM model gave 98% accuracy and BernoulliNB model gave 95.77% accuracy.
