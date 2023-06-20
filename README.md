# Introduction

I have created a bsaic machine learning algorithm using Linear regression that predicts the Salary based on the Years of Experience. For this you need to
have thre required libraries such as Pandas and Matplotlib. I have taken a dataset which contains the salary and years of experience of 30 individuals.

#Steps

**`Cleaning Data-`** Data is the most important aspect for any model to be created. Data in the right format is necessary to create efficient models with 
least amount of errors. So, upon obtaining the data, the first step is to clean it by checking for `Null values`, `Duplicates`, `Outliers` and checking the `data type` and `structure`.

**`EDA-`** To develop a linear regression model, it is imperative to have a linear relation between the Feature and Target Variable.
This can be checked by either plotting the points on a graph to manually check the relation or finding out the correlation between the feature and 
target variable. A correlation above 0.2 to 1 and below -0.2 to -1 is favourable to state liner relation between the two points.

**`Splitting Data-`** the next step is to split data into two sections so as to train the model on one and test in on the other dataset. I have used
75:25 ratio to for training and testing respectively using the `train_test_split` function from `sklearn.model_selection` library.

**`Modelling Algorith-`** We use the model y = m^x + c for Linear regression algorithms. USing the Linear Regression 
