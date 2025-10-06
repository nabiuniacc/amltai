import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from fcmeans import FCM

#User defined data preprocessing helper function:
def side_by_side(*objs, **kwds):
    space = kwds.get('space', 4)
    reprs = [repr(obj).split('\n') for obj in objs]
    # Convert list of lists to a list of strings
    reprs = ['\n'.join(i) for i in reprs] #This will convert the list of lists to a list of strings
    print ('\n'.join(reprs))
    print()
    return

mall_dataset = pd.read_csv('C:/Users/c3683414/PycharmProjects/amltai/data/Mall_Customers.csv')
print(mall_dataset.columns)

### we don't need customer id
mall_dataset.drop(['CustomerID'], axis=1, inplace=True)
mall_dataset.head()

## Gender column - label encoding
gender_mapping = {"Female": 1, "Male": 0}
mall_dataset['Gender'] = mall_dataset['Gender'].map(gender_mapping)

# out of total rows how many rows of them are Null
side_by_side(mall_dataset.isnull().sum(), mall_dataset.count())
male_customers = mall_dataset[mall_dataset.Gender == 0].shape[0]
female_customers = mall_dataset[mall_dataset.Gender == 1].shape[0]

px.pie(values=[male_customers, female_customers], names=['Male', 'Female'], title='Gender', width=600, height=400)
fig = px.box(mall_dataset, y="Spending Score (1-100)", x='Gender', width=600, height=400)
fig.show()

fig = px.histogram(mall_dataset, x="Age", color="Gender", marginal="rug", width=600, height=400)
fig.show()

fig = px.histogram(mall_dataset, x="Annual Income (k$)", color="Gender", marginal="rug", width=600, height=400)
fig.show()

fig = px.histogram(mall_dataset, x="Spending Score (1-100)", color="Gender", marginal="rug", width=600, height=400)
fig.show()

fig,ax = plt.subplots(figsize=(6,4)) ## play with size
fig.suptitle("Correlation between features", fontsize=16)
corrcoef = mall_dataset.corr()
mask = np.array(corrcoef)
mask[np.tril_indices_from(mask)] = False
sns.heatmap(corrcoef, mask=mask, vmax=.8, annot=True, ax=ax)
plt.show()

from yellowbrick.cluster import KElbowVisualizer

print('Elbow Method to determine the number of clusters to be formed:')
Elbow_M = KElbowVisualizer(KMeans(), k=14, timings=False)
Elbow_M.fit(mall_dataset)
Elbow_M.show()

number_clusters = 4
from fcmeans import FCM

fcm = FCM(n_clusters=number_clusters)
fcm.fit(mall_dataset.values)

# outputs
fcm_centers = fcm.centers
fcm_labels = fcm.predict(mall_dataset.values)

mall_dataset['Fuzzy_cluster'] = fcm_labels
mall_dataset

sns.relplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Fuzzy_cluster', size='Gender', data=mall_dataset, palette = "Paired")
plt.show()

sns.relplot(x='Age', y='Annual Income (k$)', hue='Fuzzy_cluster', size='Gender', data=mall_dataset, palette
= "Paired")
plt.show()

avg_spendingScore = mall_dataset.groupby('Fuzzy_cluster')['Spending Score (1-100)'].mean()
print(avg_spendingScore)


