# Home Equity Default Prediction using Machine Learning Techniques
![WhatsApp Image 2023-08-19 at 2 31 11 PM](https://github.com/Harshit3726/Project_on_Home_Equity_Prediction/assets/117848999/407ef00c-17b5-4a45-8590-0b4902628fab)

This Project is an Real Life Application of Machine Learning Techniques in finance field. This project aims to predict the default on home equity loans using historical data. The dataset contains information about applicants, their loan requests, and their loan performance. The goal is to build a credit scoring model that can help the bank automate the decision-making process for approving home equity lines of credit.

## Project Objective

1. **Reducing Losses**: The primary objective is to reduce losses incurred by the bank due to clients who may default on their home equity loans. By accurately identifying high-risk applicants, the bank can make more informed lending decisions.

2. **High Recall Rate**: We prioritize achieving a high recall rate to correctly identify as many clients as possible who are at risk of not paying their debts. This is crucial for risk management and minimizing potential defaults.

## Dataset Overview

- **Target Variable**: The target variable, 'BAD,' is binary, where '1' indicates that an applicant defaulted on the loan or was seriously delinquent, and '0' indicates that the applicant paid the loan on time.

- **Key Features**:
  - 'LOAN': Amount of the loan request.
  - 'MORTDUE': Amount due on the existing mortgage.
  - 'VALUE': Value of the current property.
  - 'REASON': The reason for the loan request, either debt consolidation or home improvement.
  - 'JOB': Occupational categories of applicants.
  - 'YOJ': Years at the present job.
  - 'DEROG': Number of major derogatory reports.
  - 'DELINQ': Number of delinquent credit lines.
  - 'CLAGE': Age of the oldest credit line in months.
  - 'NINQ': Number of recent credit inquiries.
  - 'CLNO': Number of credit lines.
  - 'DEBTINC': Debt-to-income ratio.

## Understanding Home Equity Loans

- **Home Equity Loan**: This type of loan uses the equity built in a home as collateral to obtain a loan. The 'LOAN,' 'MORTDUE,' and 'VALUE' columns are key factors in determining available equity and the approved loan amount.

## Understanding Defaulters

- **Defaulters**: These are individuals who fail to meet the agreed-upon terms of a loan, resulting in missed payments or total non-payment. The 'BAD' column indicates whether an applicant is a defaulter (1) or not (0).

## Models and Performance Metrics

We built three supervised classification models:
- Logistic Regression
- Support Vector Machine
- Decision Tree Classifier


The performance metric used for all models is the area under the ROC curve (AUC), which helps assess the models' ability to distinguish between defaulters and non-defaulters.

## About the Dataset
In the United States, a home equity loan is a financial product secured by a mortgage or a deed of trust. Typically, there is an existing loan on the property, often referred to as the "senior" mortgage, which takes priority in terms of repayment. Home Equity Loan (HMEQ) reports characteristics and delinquency information for 5,960 home equity loans. A home equity loan is a loan where the obligor uses the equity of his/her home as the underlying collateral.In this project, we predict the probability of default on home equity loan. The dataset contains two classes - The majority (negative) class comprises 80% of the observations and represents the applicants that paid their loan on time and 20% of the dataset is the minority (positive) class, which represents the applicants who defaulted on thier loan.
The dataset also contains few missing values in some variables, which were imputed before modeling.

## What is Machine Learning?
Machine Learning is the field of study that gives computers the capability to learn without being explicitly programmed. ML is one of the most exciting technologies that one would have ever come across. As it is evident from the name, it gives the computer that makes it more similar to humans: The ability to learn. Machine learning is actively being used today, perhaps in many more places than one would expect.

Supervised learning is when the model is getting trained on a labelled dataset. A labelled dataset is one that has both input and output parameters. In this type of learning both training and validation, datasets are labelled.

Unsupervised learning is the training of a machine using information that is neither classified nor labeled and allowing the algorithm to act on that information without guidance. Here the task of the machine is to group unsorted information according to similarities, patterns, and differences without any prior training of data.
![WhatsApp Image 2023-08-19 at 4 49 00 PM (1)](https://github.com/Harshit3726/Project_on_Home_Equity_Prediction/assets/117848999/6611077a-02f7-4068-a5d8-073d4c70373d)
## Logistic Regression
Logistic regression is basically a supervised classification algorithm. In a classification problem, the target variable(or output), y, can take only discrete values for a given set of features(or inputs), X.

Contrary to popular belief, logistic regression is a regression model. The model builds a regression model to predict the probability that a given data entry belongs to the category numbered as “1”. Just like Linear regression assumes that the data follows a linear function, Logistic regression models the data using the sigmoid function
## Sigmoid Function: g(z) = 1/1+e^-z
![WhatsApp Image 2023-08-19 at 4 56 46 PM](https://github.com/Harshit3726/Project_on_Home_Equity_Prediction/assets/117848999/38380ebf-0fa8-4155-af0e-861de9514724)
## Support Vector Machine
Support Vector Machine (SVM) is a supervised machine learning algorithm that can be used for both classification or regression challenges. However, it is mostly used in classification problems. In the SVM algorithm, we plot each data item as a point in n-dimensional space (where n is a number of features you have) with the value of each feature being the value of a particular coordinate. Then, we perform classification by finding the hyper-plane that differentiates the two classes very well.

To separate the two classes of data points, there are many possible hyperplanes that could be chosen. Our objective is to find a plane that has the maximum margin, i.e the maximum distance between data points of both classes. Maximizing the margin distance provides some reinforcement so that future data points can be classified with more confidence.
![WhatsApp Image 2023-08-19 at 5 04 27 PM](https://github.com/Harshit3726/Project_on_Home_Equity_Prediction/assets/117848999/025be9e9-5432-4bce-b4b4-001d5adc3c06)

## Decision Tree Classification Algorithm
Decision Tree is a Supervised learning technique that can be used for both classification and Regression problems, but mostly it is preferred for solving Classification problems. It is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome.

In a Decision tree, there are two nodes, which are the Decision Node and Leaf Node. Decision nodes are used to make any decision and have multiple branches, whereas Leaf nodes are the output of those decisions and do not contain any further branches.

The decisions or the test are performed on the basis of features of the given dataset.
It is a graphical representation for getting all the possible solutions to a problem/decision based on given conditions.
It is called a decision tree because, similar to a tree, it starts with the root node, which expands on further branches and constructs a tree-like structure.

In order to build a tree, we use the CART algorithm, which stands for Classification and Regression Tree algorithm.
A decision tree simply asks a question, and based on the answer (Yes/No), it further split the tree into subtrees.
Below diagram explains the general structure of a decision tree:
![WhatsApp Image 2023-08-19 at 5 22 33 PM](https://github.com/Harshit3726/Project_on_Home_Equity_Prediction/assets/117848999/ca2c043f-494e-4c05-85f9-35edfc49e8ec)

## Basic Working Pipeline of the Project
![WhatsApp Image 2023-08-19 at 5 08 59 PM](https://github.com/Harshit3726/Project_on_Home_Equity_Prediction/assets/117848999/4e27e156-3cf5-46e6-a4a4-e9b13226c53f)
The following steps are used to create a Machine Learning Project using predefined dataset:

**1. Data Collection**: The very first step is to collect the data and required dependencies.

**2. Data Analysis and Data Preprocessing**
   
**2.1 Data Analysis**: After that, we used to analyze the dataset, about it's behaviour, trend and changes with respect to independent attributes. We also have to analyze the dependencies between every attribute, such that our model must be fitted perfectly.

**2.2 Data Cleaning and Preprocessing**: This is the most important step in model creation. The Cleaning of our dataset, such that the good data can be used for next process and unrequired/bad data should be removed so that the condition of overfitting and underfitting won't occur.

**4. Feature Enginnering**: It is a machine learning technique that leverages data to create new variables that aren't in the training set. It can produce new features for both supervised and unsupervised learning, with the goal of simplifying and speeding up data transformations while also enhancing model accuracy.

**5. Data Splitting (Train-Test)**: Now we will spilt the dataset into two parts, the training part will be used to create the model and testing part will be used to verify the results of our model.

**6.** At last, we will select the most accurate result between both the techniques.

## Required Dependencies
[numpy](https://github.com/numpy/numpy)

[pandas](https://github.com/pandas-dev/pandas)

[matplotlib](https://github.com/matplotlib/matplotlib)

[scikit-learn](https://github.com/scikit-learn/scikit-learn)

[seaborn](https://github.com/seaborn/seaborn)


## Results


## Conclusion
This project provides a valuable tool for the bank's consumer credit department to make more informed lending decisions, reducing the risk of losses due to defaults. By achieving a high recall rate, the bank can proactively identify clients at risk of not paying their debts.

