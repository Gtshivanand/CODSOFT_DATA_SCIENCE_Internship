# SALES PREDICTION USING PYTHON:


# Introduction: 

The "Sales Prediction Using Advertising Costs" project aims to predict sales based on advertising expenditures across different channels like TV, radio, and digital media. By analysing historical data, the goal is to create a machine learning model that forecasts sales based on advertising budgets, helping businesses optimize marketing strategies and improve decision-making for better profitability.


# Task Description:

- Sales prediction involves forecasting the amount of a product that customers will purchase, taking into account various factors such as advertising expenditure, target audience segmentation, and advertising platform selection. 
- In businesses that offer products or services, the role of a Data Scientist is crucial for predicting future sales. They utilize machine learning techniques in Python to analyse and interpret data, allowing them to make informed decisions regarding advertising costs. By leveraging these predictions, businesses can optimize their advertising strategies and maximize sales potential. Let's embark on the journey of sales prediction using machine learning in Python.

# Problem Statement:
The goal of the "Sales Prediction Using Advertising Costs" project is to predict the sales of a company based on the amount spent on advertising across various channels, such as TV, radio, and digital media. Advertising plays a significant role in influencing consumer behavior, and businesses need to understand the relationship between advertising expenditures and sales outcomes.
Given historical data on advertising costs and corresponding sales figures, the task is to develop a machine learning model that can accurately predict future sales based on the allocated advertising budgets. This will help businesses optimize their marketing strategies, allocate budgets effectively, and make informed decisions to maximize their return on investment (ROI) in advertising.



# Objective:

The dataset has advertising data sales (in thousands of units) for a particular product advertising budgets (in thousands of dollars) for TV, radio, and newspaper media. On the basis of this data, we need to suggest a marketing plan for future sales that will result in high product sales. We have to create various regression models with a focus on robust performance. And the identify the optimal model based on to its balanced accuracy and generalization. So, we have to use this advertising dataset given in the task and analyse the predicted sales based on the given advertising expenditures using the best regression model. 

#  About the Dataset:

The advertising dataset [link](https://www.kaggle.com/code/ashydv/sales-prediction-simple-linear-regression/input) consists of sales of the products in 200 different markets. It also includes advertising budgets for the product in each of those markets for three different media: TV, radio, and newspapers. The data frame with 200 rows and 4 variables are as follows:

- TV: a numeric vector indicating the advertising budget on TV.
- Radio: a numeric vector indicating the advertising budget on radio.
- Newspaper: a numeric vector indicating the advertising budget on newspaper.
- Sales: a numeric vector indicating the sales of the interest product.




# Overview:
- The dataset has advertising data sales (in thousands of units) for a particular product advertising budgets (in thousands of dollars) for TV, radio, and newspaper media.
- On the basis of this data, we need to suggest a marketing plan for future sales that will result in high product sales.
- We have to create various regression models with a focus on robust performance
- Utilized the best regression model by using the advertising dataset given in the task and analysed the predicted sales based on the given advertising expenditures 
- Implemented the project using Python with Pandas, NumPy, Scikit-learn, and Seaborn for analysis and model development.

## Project Details:
- Objective: Predicting sales based on the given advertising expenditures
- Model Used: KNN Regression, Decision Tree Regression, ElasticNet Regression, Lasso Regression, Linear Regression, Ridge Regression, XGBoost Regression, Random Forest Regression and Gradient Boosting from Scikit-learn
- Best Model: Gradient Boosting Regression
- Accuracy Achieved: 96%
- MSE achieved:  1.245

## Key Features:
- Analysed the Sales Prediction Dataset, navigated through data visualization, preprocessing, and machine learning model selection.
- The characteristics of the dataset, including the nature of relationships and noise, influence model performance. 
- Observed outliers in the Newspapers category, while the other categories have no outliers. 
- Among the models evaluated, Gradient Boosting Regression model performed the best with around 96% accuracy and lowest Mean Squared Error (MSE) of 1.245, followed by Random Forest Regression model with 95% accuracy and Mean Squared Error (MSE) of 1.496 and XGBoost Regression model with 93.9% accuracy and Mean Squared Error (MSE) of 1.880.
- Saved the output file with Gradient Boosting Regression model's sales predictions for deployment




# visualizations:

### Visualization of missing values:

### Visualizing outliers using boxplots:

### Visualizing outliers for the target variable:

### using scatterplot to see how sales is related to the other variables: 

### Relationship between Advertising Budgets and Sales:

### Average Spend on Advertising Channels:

### Checking correlation between variables:

###  Visualize Predicted Vs Actual Values (Linear Regression):

###  Visualize Predicted Vs Actual Values(Lasso Regression):


###  Visualize Predicted Vs Actual Values(Ridge Regression):

###  Visualize Predicted Vs Actual Values(ElasticNet Regression):

###   Plotting the pie chart R-squared Scores of Regressors:

### Visualize Predicted Vs Actual Values for (Gradient Boosting Regression):

### MSE Scores of Models:




# Conclusion:

- In analysing the Sales Prediction Dataset, I navigated through data visualization, preprocessing, and machine learning model selection.
- The characteristics of the dataset, including the nature of relationships and noise, influence model performance. Simple models like Linear Regression and Regularization methods like Lasso, Ridge are best when dealing with linear relationships.
- We have observed that there is a small number of outliers in the Newspapers category, while the other categories have no outliers. And since Newspaper category have less corelation with target, no outliers treatment is required.
- Among the models evaluated, Gradient Boosting Regression model performed the best with around 96% accuracy and lowest Mean Squared Error (MSE) of 1.245, followed by Random Forest Regression model with 95% accuracy and Mean Squared Error (MSE) of 1.496 and XGBoost Regression model with 93.9% accuracy and Mean Squared Error (MSE) of 1.880. 
- Hence, Gradient Boosting Regression is identified as the optimal model based on to its balanced accuracy and generalization with accurate sales predictions.



