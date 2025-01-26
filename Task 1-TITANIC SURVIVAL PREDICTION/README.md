#  Task 1:TITANIC SURVIVAL PREDICTION:

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org) [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io) [![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)

# Introduction :
The Titanic Survival Prediction project is a classic and widely recognized machine learning task. It involves analyzing and predicting the likelihood of survival for passengers aboard the Titanic based on various factors such as their age, gender, ticket class, and more. This project is both an excellent starting point for beginners and a challenging problem for advanced practitioners due to the real-world complexity of the dataset.

# Background:
On April 15, 1912, the Titanic, a British passenger liner, sank after hitting an iceberg during its maiden voyage. Over 1,500 passengers and crew lost their lives, making it one of the deadliest maritime disasters in modern history. The tragedy has since been a subject of intense study and analysis.

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

# visualization:
 
  ###  Exploratory Data Analysis (EDA):

### Finding the Missing values:
<img src = "https://github.com/Gtshivanand/-CODSOFT-DATA-SCIENCE-Internship/blob/main/Task%201-TITANIC%20SURVIVAL%20PREDICTION/Images/missing%20values.png"/>


### Handling the Missing values:
<img src = "https://github.com/Gtshivanand/-CODSOFT-DATA-SCIENCE-Internship/blob/main/Task%201-TITANIC%20SURVIVAL%20PREDICTION/Images/ReplacedMissing%20values.png"/>

### Distribution of Survival:
<img src = "https://github.com/Gtshivanand/-CODSOFT-DATA-SCIENCE-Internship/blob/main/Task%201-TITANIC%20SURVIVAL%20PREDICTION/Images/Distribution%20of%20Survival.png"/>

### Distribution for survival distribution by Sex:
<img src = "https://github.com/Gtshivanand/-CODSOFT-DATA-SCIENCE-Internship/blob/main/Task%201-TITANIC%20SURVIVAL%20PREDICTION/Images/Distribution%20of%20Survival%20by%20Sex.png"/>


### Distribution for Survival by Passenger Class:
<img src = "https://github.com/Gtshivanand/-CODSOFT-DATA-SCIENCE-Internship/blob/main/Task%201-TITANIC%20SURVIVAL%20PREDICTION/Images/Distribution%20for%20Survival%20by%20Passenger%20Class.png"/>

### Distribution for Survival by Port of Embarkation:
<img src = "https://github.com/Gtshivanand/-CODSOFT-DATA-SCIENCE-Internship/blob/main/Task%201-TITANIC%20SURVIVAL%20PREDICTION/Images/Distribution%20for%20Survival%20by%20Port%20of%20Embarkation.png"/>

### correlation matrix:
<img src = "https://github.com/Gtshivanand/-CODSOFT-DATA-SCIENCE-Internship/blob/main/Task%201-TITANIC%20SURVIVAL%20PREDICTION/Images/Correlation%20Heatmap.png"/>


### Model Training:
<img src = "https://github.com/Gtshivanand/-CODSOFT-DATA-SCIENCE-Internship/blob/main/Task%201-TITANIC%20SURVIVAL%20PREDICTION/Images/final_accuracy%20.png"/>

 ### 10-fold Cross Validation Results:
<img src = "https://github.com/Gtshivanand/-CODSOFT-DATA-SCIENCE-Internship/blob/main/Task%201-TITANIC%20SURVIVAL%20PREDICTION/Images/10-fold%20Cross%20Validation%20Results.png"/>


  # Conclusion:
- Our analysis unveiled key insights into the Titanic dataset. We addressed missing values by filling null entries in the Age and Fare columns with medians due to the presence of outliers, while the Cabin column was discarded due to huge amount of null values.
- Notably, All the female passengers survived and all the male passengers not survived. 
- Furthermore, we observed that Passenger class 3 had the highest number of deaths and most of the Passenger class 1 have survived.
- Most of the Passengers from Queenstown had a higher survival rate compared to those from Southampton.
- In this Titanic Survival Prediction analysis, we have explored various aspects of the dataset to understand the factors influencing survival. 
- We found that only 152 passengers i.e. 36.4% of the passengers survived the crash, with significant differences in survival rates among different passenger classes, genders, and age groups. 
- The dataset also revealed that certain features, such as Fare and embarkation location, played a role in survival. 
- We trained several classification models to predict survival, most of which performed well, likely due to the relatively small dataset size. Out of which, SVM model gave 98% accuracy and BernoulliNB model gave 95.77% accuracy.
