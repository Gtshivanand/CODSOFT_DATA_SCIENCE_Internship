# CREDIT CARD FRAUD DETECTION:


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

### details of both Fraudulent and Non-Fraudulent transaction amount:

### details of both Fraudulent and Non-Fraudulent transaction time:

### Plotting the distribution of all numerical features:
### Plotting Distribution of Transaction Amount and Transaction Time:

### Visualizing the class distribution in percentage:

### Plotting Value Counts of 'Class' Column:

### Correlation Matrix:

### Plotting Balanced Classes with resampled data:

### Plotting Distribution of Features by Class after Resampling:

### Displaying the Confusion Matrix Logistic Regression Model:

### Evaluate the model by AUPRC for Logistic Regression Model:

### Displaying the Confusion Matrix Random Forest Classifier Model:

### Evaluate the model by AUPRC for Random Forest Classifier:

### Plotting ROC and AUC Curve:

### F1 Scores for both the Models:

### Comparison of Model Performance Metrics:


# Conclusion:

- Both models are effective in handling the classification task with high accuracy and balanced performance across classes. Random Forest classifier outperform the Logistic Regression in distinguishing between classes.
- Data is highly imbalanced, so it needs to be under sampling or oversampling technique. I choose to do UnderSampling technique to balance the data.
- The models show balanced performance in correctly identifying instances of both classes (0 and 1), as indicated by the similarity in f1 score and recall values.
- Precision for both classes are also high, suggesting a good balance between  and recall. But Precision for Random Forest Classifier is more than compared to Precision of Logistic Regression.
- AUC-ROC is less sensitive to class imbalance than AUC-PR. In an imbalanced dataset, where one class is much more prevalent than the other, the ROC curve may look good even if the classifier is performing poorly on the minority class.
- Depending on the specific requirements of the problem (e.g., the importance of false positives vs. false negatives), you may choose one model over the other based on the balance between precision and recall.
- Consider the context of application and certain misclassifications which may tend to high cost than others when selecting a final model.
- Further analysis, such as feature importance or exploring additional evaluation metrics, could provide additional insights into the model's behavior and help in making a more informed decision.



