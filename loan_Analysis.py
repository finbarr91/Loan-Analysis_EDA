import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
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
print("\nmissing values in rows:\n",loan.isnull().sum(axis=1))

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

"""Data Analysis
Let's now move to data analysis. To start with, let's understand the objective of the analysis clearly and identify the variables that we want to consider for analysis.

The objective is to identify predictors of default so that at the time of loan application, we can use those variables for approval/rejection of the loan. Now, there are broadly three types of variables - 1. those which are related to the applicant (demographic variables such as age, occupation, employment details etc.), 2. loan characteristics (amount of loan, interest rate, purpose of loan etc.) and 3. Customer behaviour variables (those which are generated after the loan is approved such as delinquent 2 years, revolving balance, next payment date etc.).

Now, the customer behaviour variables are not available at the time of loan application, and thus they cannot be used as predictors for credit approval.

Thus, going forward, we will use only the other two types of variables.
"""

behaviour_var =  [
  "delinq_2yrs",
  "earliest_cr_line",
  "inq_last_6mths",
  "open_acc",
  "pub_rec",
  "revol_bal",
  "revol_util",
  "total_acc",
  "out_prncp",
  "out_prncp_inv",
  "total_pymnt",
  "total_pymnt_inv",
  "total_rec_prncp",
  "total_rec_int",
  "total_rec_late_fee",
  "recoveries",
  "collection_recovery_fee",
  "last_pymnt_d",
  "last_pymnt_amnt",
  "last_credit_pull_d",
  "application_type"]

# Let's remove the behaviour variable from our analysis.
df = loan.drop(behaviour_var, axis=1)
print(df.info())

# also, we will not be able to use the variables zip code, address, state etc.
# the variable 'title' is derived from the variable 'purpose'
# thus let get rid of all these variables as well

df = df.drop(['title', 'url', 'zip_code', 'addr_state'], axis=1)

"""
Next, let's have a look at the target variable - loan_status. We need to relabel the values to a binary form - 0 or 1, 
1 indicating that the person has defaulted and 0 otherwise.
"""

df['loan_status'] = df['loan_status'].astype('category')
print(df['loan_status'].value_counts())

"""You can see that fully paid comprises most of the loans. The ones marked 'current' are neither fully paid not defaulted, so let's get rid of the current loans. 
Also, let's tag the other two values as 0 or 1."""

# Filtering only paid and charged off
df = df[df['loan_status'] != 'Current']
df['loan_status'] = df['loan_status'].apply(lambda x: 0 if x == 'Fully Paid' else 1)
print(df.loan_status)

# converting loan status to integer type
df['loan_status'] = df['loan_status'].apply(lambda x: pd.to_numeric(x))

# summarizing the loan status
print(df['loan_status'].value_counts())

# default rate
print(round(np.mean(df['loan_status']), 2))

#Let's first visualise the average default rates across categorical variables.

# plotting loan defaulters with the grade
sns.barplot(x='grade', y='loan_status', data =df)
plt.show()

# Lets define a function to plot the loan status against categorically variables.
def plot_cat(cat_var):
    sns.barplot(x= cat_var, y= 'loan_status', data=df)
    plt.show()

# compare default rates across grade of loan
plot_cat('grade')

# Clearly, as the grade of loan goes from A to G, the default rate increases.
# This is expected because the grade is decided by Lending Club based on the riskiness of the loan.

# term: 60 months loans default more than 36 months loans
plot_cat('term')

# sub-grade: as expected - A1 is better than A2 better than A3 and so on
plt.figure(figsize=(16, 6))
plt.xticks(rotation=45)
plot_cat('sub_grade')

# home ownership: not a great discriminator
plot_cat('home_ownership')

# verification_status: surprisingly, verified loans default more than not verifiedb
plot_cat('verification_status')

# purpose: small business loans defualt the most, then renewable energy and education
plt.figure(figsize=(16, 6))
plt.xticks(rotation=45)
plot_cat('purpose')

# let's first observe thee distribution of loans across years
# Let's first convert the year column into datetime and then extract year and month from it

print(df['issue_d'].head())

df['issue_d'] = df['issue_d'].apply(lambda x: datetime.strptime(x, '%b-%y'))

# extracting month and year from issue_date
df['month'] = df['issue_d'].apply(lambda x: x.month)
df['year'] = df['issue_d'].apply(lambda x: x.year)

# let's first observe the number of loans granted across years
print(df.groupby('year').year.count())

# number of loans across months
df.groupby('month').month.count()

# lets compare the default rates across years
# the default rate had suddenly increased in 2011, inspite of reducing from 2008 till 2010
plot_cat('year')

# lets compare the default rates across years
# the default rate had suddenly increased in 2011, inspite of reducing from 2008 till 2010
plot_cat('year')

# loan amount: the median loan amount is around 10,000
sns.displot(df['loan_amnt'])
plt.show()

"""
The easiest way to analyse how default rates vary across continous variables is to bin the variables into discrete categories.

Let's bin the loan amount variable into small, medium, high, very high.
"""
# binning loan amount
def loan_amount(n):
  if n < 5000:
    return 'low'
  elif n >= 5000 and n < 15000:
    return 'medium'
  elif n >= 15000 and n < 25000:
    return 'high'
  else:
    return 'very high'


df['loan_amnt'] = df['loan_amnt'].apply(lambda x: loan_amount(x))
print(df['loan_amnt'])

print(df['loan_amnt'].value_counts())

# let's compare the default rates across loan amount type
# higher the loan amount, higher the default rate
plot_cat('loan_amnt')

# let's also convert funded amount invested to bins
df['funded_amnt_inv'] = df['funded_amnt_inv'].apply(lambda x: loan_amount(x))

# funded amount invested
plot_cat('funded_amnt_inv')


# lets also convert interest rate to low, medium, high
# binning loan amount
def int_rate(n):
  if n <= 10:
    return 'low'
  elif n > 10 and n <= 15:
    return 'medium'
  else:
    return 'high'


df['int_rate'] = df['int_rate'].apply(lambda x: int_rate(x))

# comparing default rates across rates of interest
# high interest rates default more, as expected
plot_cat('int_rate')


# debt to income ratio
def dti(n):
  if n <= 10:
    return 'low'
  elif n > 10 and n <= 20:
    return 'medium'
  else:
    return 'high'


df['dti'] = df['dti'].apply(lambda x: dti(x))

# comparing default rates across debt to income ratio
# high dti translates into higher default rates, as expected
plot_cat('dti')


# funded amount
def funded_amount(n):
  if n <= 5000:
    return 'low'
  elif n > 5000 and n <= 15000:
    return 'medium'
  else:
    return 'high'


df['funded_amnt'] = df['funded_amnt'].apply(lambda x: funded_amount(x))

plot_cat('funded_amnt')


# installment
def installment(n):
  if n <= 200:
    return 'low'
  elif n > 200 and n <= 400:
    return 'medium'
  elif n > 400 and n <= 600:
    return 'high'
  else:
    return 'very high'


df['installment'] = df['installment'].apply(lambda x: installment(x))

# comparing default rates across installment
# the higher the installment amount, the higher the default rate
plot_cat('installment')

# annual income
def annual_income(n):
    if n <= 50000:
        return 'low'
    elif n > 50000 and n <=100000:
        return 'medium'
    elif n > 100000 and n <=150000:
        return 'high'
    else:
        return 'very high'

df['annual_inc'] = df['annual_inc'].apply(lambda x: annual_income(x))

# annual income and default rate
# lower the annual income, higher the default rate
plot_cat('annual_inc')

# employment length
# first, let's drop the missing value observations in emp length
df = df[~df['emp_length'].isnull()]

# binning the variable
def emp_length(n):
    if n <= 1:
        return 'fresher'
    elif n > 1 and n <=3:
        return 'junior'
    elif n > 3 and n <=7:
        return 'senior'
    else:
        return 'expert'

df['emp_length'] = df['emp_length'].apply(lambda x: emp_length(x))

# emp_length and default rate
# not much of a predictor of default
plot_cat('emp_length')

"""
 Segmented Univariate Analysis

We have now compared the default rates across various variables, and some of the important predictors are purpose of the loan, interest rate, annual income, grade etc.

In the credit industry, one of the most important factors affecting default is the purpose of the loan - home loans perform differently than credit cards, credit cards are very different from debt condolidation loans etc. 

This comes from business understanding, though let's again have a look at the default rates across the purpose of the loan.
"""
# purpose: small business loans defualt the most, then renewable energy and education
plt.figure(figsize=(16, 6))
plt.xticks(rotation=45)
plot_cat('purpose')

"""
In the upcoming analyses, we will segment the loan applications across the purpose of the loan, since that is a variable affecting many other variables -
 the type of applicant, interest rate, income, and finally the default rate."""
# lets first look at the number of loans for each type (purpose) of the loan
# most loans are debt consolidation (to repay otehr debts), then credit card, major purchase etc.
plt.figure(figsize=(16, 6))
sns.countplot(x='purpose', data=df)
plt.xticks(rotation=45)
plt.show()

#Let's analyse the top 4 types of loans based on purpose:
# consolidation, credit card,home improvement and major purchase.

# filtering the df for the 4 types of loans mentioned above
main_purposes = ["credit_card","debt_consolidation","home_improvement","major_purchase"]
df = df[df['purpose'].isin(main_purposes)]
print(df['purpose'].value_counts())

# plotting number of loans by purpose
sns.countplot(x=df['purpose'])
plt.xticks(rotation=45)
plt.show()

# let's now compare the default rates across two types of categorical variables
# purpose of loan (constant) and another categorical variable (which changes)

plt.figure(figsize=[10, 6])
sns.barplot(x='term', y="loan_status", hue='purpose', data=df)
plt.show()


# lets write a function which takes a categorical variable and plots the default rate
# segmented by purpose

def plot_segmented(cat_var):
  plt.figure(figsize=(10, 6))
  sns.barplot(x=cat_var, y='loan_status', hue='purpose', data=df)
  plt.show()


plot_segmented('term')

# grade of loan
plot_segmented('grade')

# home ownership
plot_segmented('home_ownership')

# In general, debt consolidation loans have the highest default rates.
# Lets compare across other categories as well.

# year
plot_segmented('year')

# emp_length
plot_segmented('emp_length')

# loan_amnt: same trend across loan purposes
plot_segmented('loan_amnt')

# interest rate
plot_segmented('int_rate')

# installment
plot_segmented('installment')

# debt to income ratio
plot_segmented('dti')

# annual income
plot_segmented('annual_inc')


"""A good way to quantify th effect of a categorical variable on default rate is to see 
'how much does the default rate vary across the categories'.
Let's see an example using annual_inc as the categorical variable."""

# variation of default rate across annual_inc
print(df.groupby('annual_inc').loan_status.mean().sort_values(ascending=False))

# one can write a function which takes in a categorical variable and computed the average
# default rate across the categories
# It can also compute the 'difference between the highest and the lowest default rate' across the
# categories, which is a decent metric indicating the effect of the varaible on default rate

def diff_rate(cat_var):
    default_rates = df.groupby(cat_var).loan_status.mean().sort_values(ascending=False)
    return (round(default_rates, 2), round(default_rates[0] - default_rates[-1], 2))

default_rates, diff = diff_rate('annual_inc')
print(default_rates)
print(diff)

"""Thus, there is a 6% increase in default rate as you go from high to low annual income. 
We can compute this difference for all the variables and roughly identify the ones that affect default rate the most."""

# filtering all the object type variables
df_categorical = df.loc[:, df.dtypes == object]
print('\n Data Frame Categorical:\n',df_categorical)
df_categorical['loan_status'] = df['loan_status']

# Now, for each variable, we can compute the incremental diff in default rates
print([i for i in df.columns])
# storing the diff of default rates for each column in a dict
d = {key: diff_rate(key)[1]*100 for key in df_categorical.columns if key != 'loan_status'}
print(d)






