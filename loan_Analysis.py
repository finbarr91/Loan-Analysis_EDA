import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
import re
loan = pd.read_csv('loan.csv', sep = ',', low_memory=False )
print('Dataset head: \n',loan.head()) # Looking at the dataset head
print('\nDataset columns \:\n:',loan.columns)
print('\nDataset Info \n', loan.info())
print('The description of the dataset:\n', loan.describe())

print(loan.isnull().sum()) # print the null values

print(round(loan.isnull().sum()/len(loan.index))*100,2)

# Removing columns having more than 90% missing values.
missing_columns = loan.columns[(loan.isnull().sum()/len(loan.index))*100 >90]
print(missing_columns)

loan = loan.drop(missing_columns, axis=1)

print (loan.shape)

print(round(loan.isnull().sum()/len(loan.index)*100),2) # to summarize the percentage of missing values again.

print("Columns with over 90 percent missing values:\n",loan.columns[loan.isnull().sum()/len(loan.index)*100>90])

# There are now 2 columns having approx 32 and 64% missing values -
# description and months since last delinquent

# let's have a look at a few entries in the columns
print(loan.loc[:,['desc', 'mths_since_last_delinq']].head())

# Sentiment analysis of the description.

"""
The column description contains the comments the applicant had written while applying for the loan. Although one can use some text analysis techniques to derive new features from this column (such as sentiment, number of positive/negative words etc.), we will not use this column in this analysis.

Secondly, months since last delinquent represents the number months passed since the person last fell into the 90 DPD group. There is an important reason we shouldn't use this column in analysis - since at the time of loan application, we will not have this data (it gets generated months after the loan has been approved), it cannot be used as a predictor of default at the time of loan approval.

Thus let's drop the two columns.
"""

# dropping the two columns
loan = loan.drop(['desc', 'mths_since_last_delinq'], axis=1)
print(loan)

# Summarize number of missing values again
print(round(loan.isnull().sum()/len(loan.index)*100,2))

# missing values in rows
print(loan.isnull().sum(axis=1))

# checking whether some rows have more 5 missing values
print(len(loan[loan.isnull().sum(axis =1)>5].index))

print(loan.info())

# The column is a character type, therefore let's convert it to float
loan['int_rate'] = loan['int_rate'].apply(lambda x: pd.to_numeric(x.split('%')[0]))

# checking the data info again
print(loan.info())
# also, lets extract the numeric part from the variable employment length

# first, let's drop the missing values from the column (otherwise the regex code below throws error)
loan = loan[~loan['emp_length'].isnull()]

# using regular expression to extract numeric values from the string

loan['emp_length'] = loan['emp_length'].apply(lambda x: re.findall('\d+', str(x))[0])

# convert to numeric
loan['emp_length'] = loan['emp_length'].apply(lambda x: pd.to_numeric(x))










