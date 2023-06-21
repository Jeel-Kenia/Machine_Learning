
# Salary Prediction Model

This project aims to build a salary prediction model based on years of experience using linear regression.

## Dataset

The dataset used for training and testing the model contains information about years of experience and corresponding salaries. The dataset has been cleaned and preprocessed.

## Preprocessing

The following preprocessing steps were performed on the dataset:
- Removed unnecessary columns (`Unnamed: 0`).
- Checked for missing values. No missing values were found.
- Checked for duplicate data. No duplicate entries were found.
- Checked for outliers. No outliers were found.

## Exploratory Data Analysis (EDA)

EDA was performed to analyze the relationship between years of experience and salary. A scatter plot was created to visualize the linear relationship between the two variables. A strong positive correlation (0.978) was observed between years of experience and salary.

## Model Training and Evaluation

The dataset was split into training and testing sets. The linear regression algorithm was used to train the model. The model achieved an R-squared score of 94%, indicating a high level of accuracy in predicting salaries based on years of experience.

## Usage

To use the model, follow these steps:

1. Install the required dependencies (e.g., scikit-learn, pandas, matplotlib).
2. Run the provided code to load the dataset, preprocess the data, train the model, and make predictions.
3. Adjust the code as needed for your specific use case, such as modifying the dataset or changing the features used for prediction.

## Dependencies

- scikit-learn
- pandas
- matplotlib

## Files

- `salary_prediction.ipynb`: Jupyter Notebook containing the code and analysis.
- `dataset.csv`: CSV file containing the preprocessed dataset.

## Results

The model successfully predicts salaries based on years of experience with a high degree of accuracy. Please refer to the Jupyter Notebook for detailed analysis and code implementation.

Feel Free to use this code and reach out to me in case of doubts or suggestions if any.
