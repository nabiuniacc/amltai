import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Change the file path location as per your settings

# Loading the Data

data = pd.read_csv('C:/Users/c3683414/PycharmProjects/amltai/data/Online Retail.csv')
print(data.head())
print(data.info())
print(data.describe())

# Exploring the columns of the data

print(data.columns)

# Exploring the different regions of transactions
print(data.Country.unique())

# Stripping extra spaces in the description
data['Description'] = data['Description'].str.strip()

# Dropping the rows without any invoice number
data.dropna(axis = 0, subset =['InvoiceNo'], inplace = True)

# Convert invoice numbers to string
data['InvoiceNo'] = data['InvoiceNo'].astype('str')

# Dropping all transactions which were done on credit
data = data[~data['InvoiceNo'].str.contains('C')]

# Transactions done in France

basket_France = (data[data['Country'] =="France"]
        .groupby(['InvoiceNo', 'Description'])['Quantity']
        .sum().unstack().reset_index().fillna(0)
        .set_index('InvoiceNo'))
print(basket_France.shape)

unique_products = data[data['Country'] =="France"]['Description'].unique()
print(unique_products)

# Transactions done in the United Kingdom

basket_UK = (data[data['Country'] =="United Kingdom"]
        .groupby(['InvoiceNo', 'Description'])['Quantity']
        .sum().unstack().reset_index().fillna(0)
        .set_index('InvoiceNo'))
print(basket_UK.shape)

unique_products = data[data['Country'] =="United Kingdom"]['Description'].unique()
print(unique_products)

# Transactions done in Portugal

basket_Por = (data[data['Country'] == "Portugal"]
              .groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('InvoiceNo'))
print(basket_Por.shape)

unique_products = data[data['Country'] =="Portugal"]['Description'].unique()
print(unique_products)

# Transactions done in Sweden

basket_Sweden = (data[data['Country'] =="Sweden"]
        .groupby(['InvoiceNo', 'Description'])['Quantity']
        .sum().unstack().reset_index().fillna(0)
        .set_index('InvoiceNo'))
print(basket_Sweden.shape)

unique_products = data[data['Country'] =="Sweden"]['Description'].unique()
print(unique_products)

# Defining the hot encoding function to make the data suitable
# for the concerned libraries
def hot_encode(x):
        if (x <= 0):
            return 0
        if (x >= 1):
            return 1

# Encoding the datasets for each country basket
#FRANCE
basket_encoded = basket_France.applymap(hot_encode)
basket_France = basket_encoded

#UK
basket_encoded = basket_UK.applymap(hot_encode)
basket_UK = basket_encoded
#PORTUGAL
basket_encoded = basket_Por.applymap(hot_encode)
basket_Por = basket_encoded
#SWEDEN
basket_encoded = basket_Sweden.applymap(hot_encode)
basket_Sweden = basket_encoded

# Building the model for France
frq_items = apriori(basket_France, min_support=0.05, use_colnames=True)

# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric="lift", min_threshold=1)

# Sort and display strongest rules
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
print(rules.head(10))

frq_items = apriori(basket_UK, min_support = 0.05, use_colnames = True)
rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
print(rules.head())

frq_items = apriori(basket_Por, min_support = 0.05, use_colnames = True)
rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
print(rules.head())

frq_items = apriori(basket_Sweden, min_support = 0.05, use_colnames = True)
rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
print(rules.head())