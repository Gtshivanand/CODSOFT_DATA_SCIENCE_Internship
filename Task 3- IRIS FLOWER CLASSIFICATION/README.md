# Task 3- IRIS FLOWER CLASSIFICATION:



# Introduction:

The Iris Flower Classification project is a popular introductory machine learning project that focuses on classifying Iris flowers into three species: Setosa, Versicolor, and Virginica, based on measurements of their sepal and petal attributes. These attributes include the sepal length, sepal width, petal length, and petal width. The dataset used in this project is widely known as the "Iris Dataset," which consists of 150 instances, with 50 instances for each of the three Iris species.


# Objective:
The main objective of this project is to develop a machine learning model that can accurately predict the species of an Iris flower based on its sepal and petal measurements. This classification problem helps in understanding how different features of a flower can be used to differentiate between species, using supervised learning algorithms.


# Task Description:

- The Iris flower dataset consists of three species: setosa, versicolor, and virginica. These species can be distinguished based on their measurements. Now, imagine that you have the measurements of Iris flowers categorized by their respective species. Your objective is to train a machine learning model that can learn from these measurements and accurately classify the Iris flowers into their respective species. 
- Use the Iris dataset to develop a model that can classify iris flowers into different species based on their sepal and petal measurements. This dataset is widely used for introductory classification tasks.


# Problem Statement: 
The objective of this project is to develop a machine learning model capable of classifying Iris flowers into one of three species—Setosa, Versicolor, or Virginica—based on the measurements of their sepals and petals. Specifically, the model will predict the species of the Iris flower based on the following features:
* Sepal Length (cm)
* Sepal Width (cm)
* Petal Length (cm)
* Petal Width (cm)


# Overview:
- This data science project focuses on classifying the Iris flower species using machine learning models.
- It includes importing necessary libraries, data inspection, data visualization, model building, generating predictions, and evaluating model performance.
- The primary goal of this multiclassification project is to develop a robust and accurate machine learning model capable of predicting the species type of an Iris flower based on its morphological features.
- The purpose of this project is to employ the required machine learning techniques to improve our comprehension of the morphological variations present within the Iris species.
- This information can be valuable for various stakeholders, including botanists, researchers, and horticulturists, as it enhances our ability to categorize and interpret the diversity present in Iris flowers.

# About the Dataset:


- The IRIS Dataset [link](https://www.kaggle.com/datasets/arshid/iris-flower-dataset) is a multivariate dataset which contains the data to quantify the morphologic variation of Iris flowers of three related species. 
- The dataset consists of 50 samples from each of three species of Iris, namely Iris Setosa, Iris virginica, and Iris versicolor. 
- Four features were measured from each sample which are the length and the width of the sepals and petals, in centimetre’s.
- The dataset is a CSV file which contains a set of 150 records under 5 attributes - Petal Length, Petal Width, Sepal Length, Sepal width and Class(Species).


## Project Details
- Objective: IRIS Flower Classification using ML Models
- Model Used: Logistic Regression, K-Nearest Neighbours (KNN) classification, Gaussian Naive Bayes classification, Support Vector Machine, Decision Tree, and Random Forest from Scikit-learn.
- Best Model: K-Nearest Neighbours (KNN)
- Accuracy achieved: 97%
- Precision: 97.8%
- Recall: 97.3%
- F1-Score: 97.4%

## Key Insights
- The dataset is balanced i.e. equal records are present for all three species.
- We have four numerical columns while just one categorical column which in turn is our target column.
- We have performed Explorative Data Analysis on the Iris dataset and created various colourful visualizations, including boxplots, pair plots, histogram, histogram with a density plot, swarm Plots, violin plots, a correlation heatmap, and a pie chart for species distribution.
- These visualizations helped in understanding the relationships and differences between Iris species and features, making it easier to classify and analyse these flowers based on their measurements.
- The setose species is the most easily distinguishable because of its small feature size. The Versicolor and Virginica species are usually mixed and are sometimes hard to separate, while usually Versicolor has average feature sizes and virginica has larger feature sizes.
- EDA with visualizations of the Iris dataset reveals that features like petal length and petal width are highly discriminative for distinguishing between the three Iris species. This information is crucial to build a machine learning model for Iris species classification.
- By employing machine learning algorithms such as Logistic Regression, K-Nearest Neighbours (KNN) classification, Gaussian Naive Bayes classification, Support Vector Machine, Decision Tree, and Random Forest, we successfully constructed models capable of precisely classifying Iris flowers into their respective species based on these measurements.
- Furthermore, the evaluation of machine learning models revealed that K-Nearest Neighbours (KNN) outperformed other classification models, with an impressive accuracy of 97% on unseen data, highlighting the model's robustness.
- Our results highlight the effectiveness of machine learning techniques in automating species classification based on morphological attributes, contributing to the fields of botany, ecology, and beyond.


# Visualization: 

### Visualization of missing values:
<img src ="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%203-%20IRIS%20FLOWER%20CLASSIFICATION/Images/missing%20values.png"/>

### Create a boxplot for each attribute excluding the target variable:
<img src ="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%203-%20IRIS%20FLOWER%20CLASSIFICATION/Images/Boxplot%20of%20Outliers%20for%20Each%20Attribute.png"/>

### Create a boxplot for all the attributes with target variable :
<img src ="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%203-%20IRIS%20FLOWER%20CLASSIFICATION/Images/Box%20Plot%20of%20attribute.png"/>

### Boxplot after replacing the outliers with the median:
<img src ="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%203-%20IRIS%20FLOWER%20CLASSIFICATION/Images/Boxplot%20after%20replacing%20the%20outliers%20with%20the%20median.png"/>

### Distribution of Species:
<img src ="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%203-%20IRIS%20FLOWER%20CLASSIFICATION/Images/Distribution%20of%20Species.png"/>

### Pair plot to visualize relationships between features:
<img src ="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%203-%20IRIS%20FLOWER%20CLASSIFICATION/Images/species.png"/>

### Distribution of data for the various columns using Histogram:
<img src ="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%203-%20IRIS%20FLOWER%20CLASSIFICATION/Images/Distribution%20of%20Sepal%20Length%20vs%20Petal%20Width.png"/>

###  Histogram to visualize relationships between variables:
<img src ="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%203-%20IRIS%20FLOWER%20CLASSIFICATION/Images/distribution%20by%20each%20species.png"/>

### Histogram with a density plot to visualize relationships between variables:
<img src ="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%203-%20IRIS%20FLOWER%20CLASSIFICATION/Images/Attributes.png"/>

### Swarm plots for each attribute:
 <img src ="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%203-%20IRIS%20FLOWER%20CLASSIFICATION/Images/Attribute2.png"/>

###  Violin plot to visualize distribution and density by species:
<img src ="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%203-%20IRIS%20FLOWER%20CLASSIFICATION/Images/AttributesBoxplot.png"/>

###  Generic plot for Sepal length vs width and Petal length vs width:
<img src ="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%203-%20IRIS%20FLOWER%20CLASSIFICATION/Images/Lenght%20vs%20Width.png"/>

###  Visualising  the correlation using a heatmap:
<img src ="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%203-%20IRIS%20FLOWER%20CLASSIFICATION/Images/Correlation%20Heatmap.png"/>

### Distribution of Species after encoding:
<img src ="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%203-%20IRIS%20FLOWER%20CLASSIFICATION/Images/Distribution%20of%20Species%20after%20encoding.png"/>

###  Displaying the Confusion Matrix for Logistic Regression Model:
<img src ="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%203-%20IRIS%20FLOWER%20CLASSIFICATION/Images/Logistic%20Regression%20model%20%20Heatmap.png"/>

###  Displaying the Confusion Matrix for K-Nearest Neighbours Model:
<img src ="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%203-%20IRIS%20FLOWER%20CLASSIFICATION/Images/K-Nearest%20Neighbors%20Model%20Heatmap.png"/>

###  Displaying the Confusion Matrix for  Gaussian Naive Bayes Model:
<img src ="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%203-%20IRIS%20FLOWER%20CLASSIFICATION/Images/Gaussian%20Naive%20Bayes%20Model%20Heatmap.png"/>

###  Displaying the Confusion Matrix for  Support Vector Machine Model:
<img src ="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%203-%20IRIS%20FLOWER%20CLASSIFICATION/Images/SVM_model%20Heatmap.png"/>

###  Displaying the Confusion Matrix for Decision Tree Model:
<img src ="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%203-%20IRIS%20FLOWER%20CLASSIFICATION/Images/DTree_model%20Heatmap.png"/>

### Decision Tree Model:
<img src ="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%203-%20IRIS%20FLOWER%20CLASSIFICATION/Images/DTree_model.png"/>

###  Displaying the Confusion Matrix for Random Forest Model:
<img src ="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%203-%20IRIS%20FLOWER%20CLASSIFICATION/Images/Random%20Forest%20Model%20Heatmap.png"/>

### Random Forest Model:
<img src ="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%203-%20IRIS%20FLOWER%20CLASSIFICATION/Images/RF_model.png"/>

### Model Accuracy Comparison:
<img src ="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%203-%20IRIS%20FLOWER%20CLASSIFICATION/Images/Model%20Accuracy%20Comparison.png"/>

### Predicted Species Distribution:
<img src ="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%203-%20IRIS%20FLOWER%20CLASSIFICATION/Images/Predicted%20Species%20Distribution.png"/>

### Distribution Predicted Species:
<img src ="https://github.com/Gtshivanand/CODSOFT_DATA_SCIENCE_Internship/blob/main/Task%203-%20IRIS%20FLOWER%20CLASSIFICATION/Images/Distribution%20Predicted%20Species.png"/>

# Conclusion:

- The dataset is balanced i.e. equal records are present for all three species.
- We have four numerical columns while just one categorical column which in turn is our target column.
- We have performed Explorative Data Analysis on the Iris dataset and created various colourful visualizations, including boxplots, pair plots, histogram,  histogram with a density plot, swarm Plots, violin plots, a correlation heatmap, and a pie chart for species distribution.
- These visualizations helped in understanding the relationships and differences between Iris species and features, making it easier to classify and analyse these flowers based on their measurements.
- The setosa species is the most easily distinguishable because of its small feature size. The Versicolor and Virginica species are usually mixed and are sometimes hard to separate, while usually Versicolor has average feature sizes and virginica has larger feature sizes.
- EDA with visualizations of the Iris dataset reveals that features like petal length and petal width are highly discriminative for distinguishing between the three Iris species. This information is crucial to build a machine learning model for Iris species classification.
- By employing machine learning algorithms such as Logistic Regression, K-Nearest Neighbours (KNN) classification, Gaussian Naive Bayes classification, Support Vector Machine, Decision Tree, and Random Forest, we successfully constructed models capable of precisely classifying Iris flowers into their respective species based on these measurements.
- Furthermore, the evaluation of machine learning models revealed that K-Nearest Neighbours (KNN) outperformed other classification models, with an impressive accuracy of 97% on unseen data, highlighting the model's robustness. 
- Our results highlight the effectiveness of machine learning techniques in automating species classification based on morphological attributes, contributing to the fields of botany, ecology, and beyond.




