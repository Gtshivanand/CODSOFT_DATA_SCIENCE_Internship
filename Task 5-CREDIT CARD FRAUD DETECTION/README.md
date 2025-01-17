# CREDIT CARD FRAUD DETECTION:

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org) [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io) [![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)

# Introduction:
Credit card fraud detection aims to identify fraudulent transactions in real-time, protecting both consumers and financial institutions. Machine learning techniques, such as decision trees and neural networks, are used to analyse transaction patterns and detect anomalies, distinguishing between legitimate and fraudulent activity.

# Task Description:

- Build a machine learning model to identify fraudulent credit card transactions.
- Preprocess and normalize the transaction data, handle class imbalance issues, and split the dataset into training and testing sets.
- Train a classification algorithm, such as logistic regression or random forests, to classify transactions as fraudulent or genuine.
- Evaluate the model's performance using metrics like precision, recall, and F1-score, and consider techniques like oversampling or under sampling for improving results.



# Problem Statement:
Credit card fraud is a growing concern in the financial industry, leading to significant financial losses and impacting customer trust. Detecting fraudulent transactions in real-time is a major challenge due to the large volume of transactions and the evolving tactics used by fraudsters. The problem is to develop an efficient machine learning model that can accurately classify credit card transactions as either legitimate or fraudulent. The model should be capable of identifying fraudulent activities with high precision, minimizing false positives, and ensuring scalability to handle large datasets in real-time.


# About the Dataset:

- The Credit Card Fraud Detection dataset [link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) comprises credit card transactions made by European cardholders in September 2013. 
- It covers two days, featuring 492 frauds out of 284,807 transactions. Notably, the dataset is highly unbalanced, with fraudulent transactions accounting for only 0.172% of all transactions.
- This dataset exclusively includes numerical input variables resulting from a PCA transformation. Due to confidentiality constraints, the original features and additional background information aren't provided. 
- The features V1 through V28 represent principal components obtained via PCA. However, 'Time' and 'Amount' are the only features not subjected to PCA.
    - 'Time' indicates the seconds elapsed between each transaction and the first recorded transaction.
    - 'Amount' signifies the transaction amount, potentially useful for example-dependent cost-sensitive learning.
    - 'Class' represents the response variable, assuming a value of 1 for fraud and 0 otherwise.




## Overview
- Developed a Fraudulent transactions detection system to prevent customers from being charged for unauthorized purchases using machine learning on credit card transaction data.
- Data is highly imbalanced, so it needs to be under sampling or oversampling technique. 
- Implemented the project using Python with Pandas, NumPy, Scikit-learn, and Seaborn for analysis and model development.
- Utilized Logistic Regression Model and Random Forest Classifier and achieved more than 94% accuracy on test data for both the models
- Both models (Logistic Regression and Random Forest Classifier) performed exceptionally well, but Random Forest Classifier have achieved high F1-score when compared to Logistic Regression model which makes it the best model.
- Performed both logistic regression and Random Forest Classifier using scikit-learn, made predictions on a test set, and plotted the confusion matrix for evaluation.
- Evaluate the model's performance using metrics like precision, recall, and F1-score.
- Plotted ROC-AUC curve and Precision-Recall curve to decide the proper threshold values and to explain the model's goodness of fit.

## Project Details
- Objective: Credit Card Fraud Detection using Machine Learning techniques.
- Model Used: Logistic Regression and Random Forest Classifier from Scikit-learn.
- Accuracy Achieved for both models: >94%.
- F1 score for Random Forest Classifier: 95%
- F1 score for Logistic Regression model: 94%

## Key Features
- Utilized Logistic Regression and Random Forest Classifier to predict fraudulent transactions based on transaction features.
- Focused on minimizing false positives and ensuring model interpretability.
- Data is highly imbalanced, so it needs to be under sampling or oversampling technique. I choose to do UnderSampling technique to balance the data.
- The models show balanced performance in correctly identifying instances of both classes (0 and 1), as indicated by the similarity in f1 score and recall values.
- Precision for both classes are also high, suggesting a good balance between and recall. But Precision for Random Forest Classifier is more than compared to Precision of Logistic Regression.


# Visualization:

### Visualization of missing values:

<img src="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%205-CREDIT%20CARD%20FRAUD%20DETECTION/Images/missing%20values.png"/>

### details of both Fraudulent and Non-Fraudulent transaction amount:

<img src="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%205-CREDIT%20CARD%20FRAUD%20DETECTION/Images/Trans_Amount.png"/>

### details of both Fraudulent and Non-Fraudulent transaction time:

<img src="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%205-CREDIT%20CARD%20FRAUD%20DETECTION/Images/Trans_time.png"/>

### Plotting the distribution of all numerical features:

<img src="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%205-CREDIT%20CARD%20FRAUD%20DETECTION/Images/ccfd_df.png"/>

### Plotting Distribution of Transaction Amount and Transaction Time:

<img src="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%205-CREDIT%20CARD%20FRAUD%20DETECTION/Images/Distribution%20of%20Transaction%20Amount%20%26%20Time.png"/>

### Visualizing the class distribution in percentage:

<img src="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%205-CREDIT%20CARD%20FRAUD%20DETECTION/Images/Class%20Distribution.png"/>

### Plotting Value Counts of 'Class' Column:

<img src="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%205-CREDIT%20CARD%20FRAUD%20DETECTION/Images/Value%20Counts%20of%20column.png"/>

### Correlation Matrix:

<img src="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%205-CREDIT%20CARD%20FRAUD%20DETECTION/Images/correlation%20matrix.png"/>

### Plotting Balanced Classes with resampled data:

<img src="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%205-CREDIT%20CARD%20FRAUD%20DETECTION/Images/Value%20Counts%20of%20class%20Column.png"/>

### Plotting Distribution of Features by Class after Resampling:

<img src="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%205-CREDIT%20CARD%20FRAUD%20DETECTION/Images/resampling.png"/>

### Displaying the Confusion Matrix Logistic Regression Model:

<img src="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%205-CREDIT%20CARD%20FRAUD%20DETECTION/Images/Confusion%20Matrix.png"/>

### Evaluate the model by AUPRC for Logistic Regression Model:

<img src="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%205-CREDIT%20CARD%20FRAUD%20DETECTION/Images/Precision-Recall%20Curve%20for%20Logistic%20Regression%20Model.png"/>

### Displaying the Confusion Matrix Random Forest Classifier Model:

<img src="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%205-CREDIT%20CARD%20FRAUD%20DETECTION/Images/Confusion%20Matrix2.png"/>

### Evaluate the model by AUPRC for Random Forest Classifier:

<img src="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%205-CREDIT%20CARD%20FRAUD%20DETECTION/Images/Precision-Recall%20Curve%20for%20Random%20Forest%20Classifier.png"/>

### Plotting ROC and AUC Curve:

<img src="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%205-CREDIT%20CARD%20FRAUD%20DETECTION/Images/ROC%20Curve%20for%20Fraud%20Detection%20Models.png"/>

### F1 Scores for both the Models:

<img src="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%205-CREDIT%20CARD%20FRAUD%20DETECTION/Images/F1%20Scores%20for%20both%20the%20Models.png"/>

### Comparison of Model Performance Metrics:

<img src="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%205-CREDIT%20CARD%20FRAUD%20DETECTION/Images/Comparison%20of%20Model%20Performance%20Metrics.png"/>

# Conclusion:

- Both models are effective in handling the classification task with high accuracy and balanced performance across classes. Random Forest classifier outperform the Logistic Regression in distinguishing between classes.
- Data is highly imbalanced, so it needs to be under sampling or oversampling technique. I choose to do UnderSampling technique to balance the data.
- The models show balanced performance in correctly identifying instances of both classes (0 and 1), as indicated by the similarity in f1 score and recall values.
- Precision for both classes are also high, suggesting a good balance between  and recall. But Precision for Random Forest Classifier is more than compared to Precision of Logistic Regression.
- AUC-ROC is less sensitive to class imbalance than AUC-PR. In an imbalanced dataset, where one class is much more prevalent than the other, the ROC curve may look good even if the classifier is performing poorly on the minority class.
- Depending on the specific requirements of the problem (e.g., the importance of false positives vs. false negatives), you may choose one model over the other based on the balance between precision and recall.
- Consider the context of application and certain misclassifications which may tend to high cost than others when selecting a final model.
- Further analysis, such as feature importance or exploring additional evaluation metrics, could provide additional insights into the model's behavior and help in making a more informed decision.



