#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Replace 'path_to_diabetes_data' with the actual path to your dataset file
diabetes_data = pd.read_csv('patients.csv')

# Display basic information about the dataset
print(diabetes_data.info())

# Display summary statistics
print(diabetes_data.describe())

# Check for missing values
print(diabetes_data.isnull().sum())

# Check the first few rows of the dataset
print(diabetes_data.head())

# Check unique values in the 'Outcome' column (assuming it's the target variable)
print(diabetes_data['Outcome'].value_counts())


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Check for any missing values
missing_values = diabetes_data.isnull().sum()
print("Missing Values:")
print(missing_values)

# Summary statistics
print("\nSummary Statistics:")
print(diabetes_data.describe())

# Data types and non-null counts
print("\nData Types and Non-Null Counts:")
print(diabetes_data.info())


# In[5]:


sns.countplot(data=diabetes_data, x='Outcome')
plt.title('Distribution of Diabetes Outcome')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.show()


# In[6]:


plt.figure(figsize=(10, 8))
sns.heatmap(diabetes_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# In[7]:


plt.figure(figsize=(10, 6))
important_columns = ['Glucose', 'BMI', 'Age', 'Insulin', 'BloodPressure']
sns.boxplot(data=diabetes_data[important_columns])
plt.title('Boxplot of Important Features')
plt.xlabel('Features')
plt.ylabel('Values')
plt.show()


# In[8]:


sns.pairplot(diabetes_data[['Glucose', 'BMI', 'Age', 'Insulin', 'BloodPressure', 'Outcome']], hue='Outcome')
plt.title('Pairplot of Select Features')
plt.show()


# In[13]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


Q1 = diabetes_data['BMI'].quantile(0.25)
Q3 = diabetes_data['BMI'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Apply outlier handling
diabetes_data['BMI'] = diabetes_data['BMI'].apply(lambda x: upper_bound if x > upper_bound else lower_bound if x < lower_bound else x)


print("Changes in 'BMI' column after handling outliers:")
print(diabetes_data[['BMI']].head())


# In[16]:


from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt

# Assuming 'diabetes_data' contains your diabetes dataset
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = diabetes_data[features]

# K-Means Clustering
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
kmeans.fit(X)
diabetes_data['KMeans_Cluster'] = kmeans.labels_

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
diabetes_data['DBSCAN_Cluster'] = dbscan.fit_predict(X)

# Visualize the clusters (you can adapt this based on your data dimensions)
plt.scatter(diabetes_data['Glucose'], diabetes_data['BMI'], c=diabetes_data['KMeans_Cluster'], cmap='viridis', marker='o', alpha=0.5)
plt.xlabel('Glucose')
plt.ylabel('BMI')
plt.title('KMeans Clustering')
plt.show()

plt.scatter(diabetes_data['Glucose'], diabetes_data['BMI'], c=diabetes_data['DBSCAN_Cluster'], cmap='viridis', marker='o', alpha=0.5)
plt.xlabel('Glucose')
plt.ylabel('BMI')
plt.title('DBSCAN Clustering')
plt.show()

#KMeans is a clustering algorithm that partitions data into K distinct clusters based on similarity, aiming to minimize the within-cluster variance.

#DBSCAN is a density-based clustering algorithm that identifies clusters as regions of high density separated by regions of low density, suitable for various shapes and sizes of clusters.


# In[25]:


print(diabetes_data)


# In[26]:


diabetes_data.drop('cluster', axis=1, inplace=True)


# In[27]:


print(diabetes_data)


# In[30]:


from sklearn.manifold import TSNE

# Assuming df is your DataFrame with features
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = diabetes_data[features]


tsne = TSNE(n_components=2)
components = tsne.fit_transform(X)
df_tsne = pd.DataFrame(data=components, columns=['TSNE1', 'TSNE2'])
print(df_tsne)

#t-SNE is a technique that compresses high-dimensional data into a lower-dimensional space (like 2D or 3D) to reveal underlying patterns or clusters while preserving local relationships among data points. It's commonly used for visualizing complex datasets in a more interpretable form.


# In[31]:


unique_age_values = diabetes_data['Age'].unique()
print(unique_age_values)


# In[32]:


unique_pregnancies_values = diabetes_data['Pregnancies'].unique()
print(unique_pregnancies_values)


# In[35]:


from sklearn.decomposition import FastICA
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = diabetes_data[features]
ica = FastICA(n_components=3)
components = ica.fit_transform(X)
df_ica = pd.DataFrame(data=components, columns=['IC1', 'IC2', 'IC3'])
print(df_ica)
#ICA decomposes original features into statistically independent components, aiding in uncovering underlying factors. Splitting into IC1, IC2, IC3 helps preserve these independent components for detailed analysis.


# In[ ]:




