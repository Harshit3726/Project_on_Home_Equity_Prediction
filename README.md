# Home Equity Default Prediction using Machine Learning Techniques
![WhatsApp Image 2023-08-19 at 2 31 11 PM](https://github.com/Harshit3726/Project_on_Home_Equity_Prediction/assets/117848999/407ef00c-17b5-4a45-8590-0b4902628fab)

This Project is an Real Life Application of Machine Learning Techniques in finance field. This project aims to predict the probability of default on home equity loans using historical data. The dataset contains information about applicants, their loan requests, and their loan performance. The goal is to build a credit scoring model that can help the bank automate the decision-making process for approving home equity lines of credit.

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
Home Equity Loan (HMEQ) reports characteristics and delinquency information for 5,960 home equity loans. A home equity loan is a loan where the obligor uses the equity of his/her home as the underlying collateral.

In this project, we predict the probability of default on home equity loan. The dataset contains two classes - The majority (negative) class comprises 80% of the observations and represents the applicants that paid their loan on time and 20% of the dataset is the minority (positive) class, which represents the applicants who defaulted on thier loan.

The dataset also contains few missing values in some variables, which were imputed before modeling. We built four supervised classification models: Logistic regression, Support vector machine, Random forest, and XGBoost. The area under the ROC curve (AUC) was used as the performance metric for all the models.



## Conclusion

This project provides a valuable tool for the bank's consumer credit department to make more informed lending decisions, reducing the risk of losses due to defaults. By achieving a high recall rate, the bank can proactively identify clients at risk of not paying their debts.

For more details, please refer to the project's code and documentation.

---

**Note**: The dataset may contain some missing values, which were imputed before modeling. The code and detailed analysis can be found in the project's repository.

