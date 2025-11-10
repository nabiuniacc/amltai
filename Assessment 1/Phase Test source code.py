import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from fcmeans import FCM

#user defined pre processing helper function:
def side_by_side(*objs, **kwds):
    space = kwds.get('space', 4)
    reprs=[repr(obj).split('\n')[0] for obj in objs]
    #covert list of lsit to str
    reprs = ['\n'.join(i) for i in reprs]
    print('\n'.join(reprs))
    print()
    return

edu_dataset = pd.read_csv('C:/Users/c3683414/PycharmProjects/amltai/data/data_.csv')
print(edu_dataset.head())
print(edu_dataset.columns)

#columns - label encoding
gender_mapping = {"Female":1, "Male":0}
edu_dataset['Gender'] = edu_dataset['Gender'].map(gender_mapping)

maritalStatus_mapping = {"Married":1, "Unmarried":0}
edu_dataset['Marital status'] = edu_dataset['Marital status'].map(maritalStatus_mapping)

specialNeeds_mapping = {"Not required":0, "required":1}
edu_dataset['Educational special needs'] = edu_dataset['Educational special needs'].map(specialNeeds_mapping)

scholarship_mapping = {"No":0, "Yes":1}
edu_dataset['Scholarship holder'] = edu_dataset['Scholarship holder'].map(scholarship_mapping)

international_mapping = {"No":0, "Yes":1}
edu_dataset['International'] = edu_dataset['International'].map(international_mapping)

#data preprocessing for null values
side_by_side(edu_dataset.isnull().sum(), edu_dataset.count())
print(edu_dataset.isnull().sum())
print(edu_dataset.count("rows"))
edu_dataset.dropna(inplace=True)
print(edu_dataset.isnull().sum())
print(edu_dataset.count("rows"))

male_students = edu_dataset[edu_dataset.Gender == 0].shape[0]
female_students = edu_dataset[edu_dataset.Gender == 1].shape[0]
married_students = edu_dataset[edu_dataset['Marital status'] == 1].shape[0]
unmarried_students = edu_dataset[edu_dataset['Marital status'] == 0].shape[0]
sen_students = edu_dataset[edu_dataset['Educational special needs'] == 1].shape[0]
nonSen_students = edu_dataset[edu_dataset['Educational special needs'] == 0].shape[0]
scholarship_student = edu_dataset[edu_dataset['Scholarship holder'] == 1].shape[0]
NScholar_student = edu_dataset[edu_dataset['Scholarship holder'] == 0].shape[0]
international_student = edu_dataset[edu_dataset['International'] == 1].shape[0]
home_student = edu_dataset[edu_dataset['International'] == 0].shape[0]

#visualizations for exploratory data analysis
fig = px.pie(values= [male_students,female_students], names=['Male Students', 'Female Students'], title= 'Gender', width= 600, height=400)
fig.show()

#boxplots
fig = px.box(edu_dataset, y= 'Unemployment rate', x='Gender', width= 600, height=400)
fig.show()
fig = px.box(edu_dataset, y= 'Age at enrollment', x='Gender', width= 600, height=400)
fig.show()
fig = px.box(edu_dataset, y= 'Admission grade', x='Gender', width= 600, height=400)
fig.show()
fig = px.box(edu_dataset, y= 'Curricular units 1st sem (grade)', x='Gender', width= 600, height=400)
fig.show()
fig = px.box(edu_dataset, y= 'Curricular units 2nd sem (grade)', x='Gender', width= 600, height=400)
fig.show()
fig = px.box(edu_dataset, y= 'GDP', x='Gender', width= 600, height=400)
fig.show()

#histograms
fig = px.histogram(edu_dataset, x='Age at enrollment', color='Gender', marginal= "rug", width= 600, height=400)
fig.show()
fig = px.histogram(edu_dataset, x='Academic Status', color='Gender', marginal= "rug", width= 600, height=400)
fig.show()
fig = px.histogram(edu_dataset, x='Academic Status', color='Marital status', marginal= "rug", width= 600, height=400)
fig.show()
fig = px.histogram(edu_dataset, x='Academic Status', color='Educational special needs', marginal= "rug", width= 600, height=400)
fig.show()
fig = px.histogram(edu_dataset, x='Curricular units 1st sem (grade)', color='', marginal= "rug", width= 600, height=400)
fig.show()

