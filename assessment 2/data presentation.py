import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
studentInfo = pd.read_csv('C:/Users/Fatima Choudhury/amltai/data/studentInfo.csv')
#data dimensionality - the shape and structure of the dataset
print("Dataset head")
print(studentInfo.head(10))
print("Dataset information:")
print(studentInfo.info())
print("NULL VALUES:")
print(studentInfo.isna().sum())
print("Duplicate summary")
print(studentInfo.duplicated().sum())
#number of rows and columns
print("Shape (rows, columns):")
print(studentInfo.shape)
#the target variable of this dataset is final_result
#describe its distribution (hist for numerical target, but probs bar chart for categorical target)
studentInfo['final_result'].value_counts().plot(kind='bar')
plt.title('Distribution of Final_Results')
plt.xlabel('Final_Results')
plt.ylabel('Count')
plt.show()
#descriptive stats
#most common categories (cat vars)
print("most common categories")
print(studentInfo.select_dtypes('object').mode().iloc[0])
#mean, median, mode, min, max (num vars)
print("Min Values in the Distribution")
print(studentInfo.select_dtypes('number').min())
print("*******************************")
print("Max Values in the Distribution")
print(studentInfo.select_dtypes('number').max())
print("*******************************")
print("Mean Values in the Distribution")
print(studentInfo.select_dtypes('number').mean())
print("*******************************")
print("Median Values in the Distribution")
print(studentInfo.select_dtypes('number').median())
print("*******************************")
print("Mode Values in the Distribution ")
print(studentInfo.select_dtypes('number').mode())


#table summarising columns

test_v = "kate"