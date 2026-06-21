#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[4]:


import pandas as pd

housing_data = pd.read_csv('housing_data.csv')
print(housing_data.head())


# In[5]:


print(housing_data.info())
print(housing_data.describe())


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(housing_data[['average_price', 'houses_sold', 'no_of_crimes']])
plt.show()


# In[26]:


import seaborn as sns
import matplotlib.pyplot as plt

# Selecting only numerical columns for correlation calculation
numerical_columns = housing_data.select_dtypes(include='number').columns.tolist()

# Calculate the correlation matrix using numerical columns
correlation_matrix = housing_data[numerical_columns].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# In[9]:


plt.figure(figsize=(10, 6))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(1, len(numerical_columns), i)
    sns.boxplot(y=housing_data[column])
    plt.title(f'Boxplot of {column}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(1, len(numerical_columns), i)
    sns.histplot(housing_data[column], kde=True)
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.show()


# In[10]:


# Visualize distributions of numerical columns
plt.figure(figsize=(12, 6))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(1, len(numerical_columns), i)
    sns.histplot(housing_data[column], kde=True)
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.show()


# In[14]:


encoded_data = pd.get_dummies(housing_data, columns=['area', 'borough_flag'])
print(encoded_data.head())


# In[15]:


from sklearn.preprocessing import MinMaxScaler

# Assuming 'average_price', 'houses_sold', and 'no_of_crimes' need scaling
columns_to_scale = ['average_price', 'houses_sold', 'no_of_crimes']
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(housing_data[columns_to_scale])
housing_data[columns_to_scale] = scaled_data
print(housing_data.head())


# In[22]:


from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

numerical_columns = ['average_price', 'houses_sold', 'no_of_crimes']

# Impute missing values in numerical columns
imputer = SimpleImputer(strategy='mean')  # You can choose 'mean', 'median', or 'most_frequent'
housing_data[numerical_columns] = imputer.fit_transform(housing_data[numerical_columns])

# Apply PCA on imputed numerical columns
pca = PCA(n_components=3)  # Specify number of components
pca_result = pca.fit_transform(housing_data[numerical_columns])
housing_data_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3'])

print(housing_data_pca.head())  # Output the transformed data


# In[27]:


sns.boxplot(x='borough_flag', y='houses_sold', data=housing_data)
plt.title('Box Plot of Houses Sold by Borough Flag')
plt.show()


# In[28]:


# Checking for missing values
missing_values = housing_data.isnull().sum()
print("Missing Values:\n", missing_values)

# Checking for duplicated rows
duplicates = housing_data.duplicated().sum()
print("\nNumber of Duplicated Rows:", duplicates)

# Checking for outliers or unrealistic values
# You might define certain thresholds or use domain knowledge to identify bad data

# Summary statistics to identify outliers
summary_statistics = housing_data.describe()
print("\nSummary Statistics:\n", summary_statistics)

# Checking for unique values in categorical columns
categorical_columns = housing_data.select_dtypes(include='object').columns.tolist()
for col in categorical_columns:
    unique_values = housing_data[col].unique()
    print(f"\nUnique values in '{col}':\n", unique_values)


# In[29]:


import pandas as pd

# Convert 'date' column to datetime format
housing_data['date'] = pd.to_datetime(housing_data['date'])

# Check consistency between 'code' and 'area' columns
code_area_check = housing_data.groupby('code')['area'].nunique()
inconsistent_entries = code_area_check[code_area_check > 1]
if len(inconsistent_entries) > 0:
    print("Inconsistent 'code' and 'area' entries:", inconsistent_entries)

# Feature engineering from 'date' column (e.g., extracting month, year)
housing_data['year'] = housing_data['date'].dt.year
housing_data['month'] = housing_data['date'].dt.month

# Outlier detection and handling - perform based on domain knowledge
# Scaling/Normalization of numerical columns if required

# Display the modified DataFrame
print(housing_data.head())


# In[30]:


# Find the conflicting entries between 'code' and 'area'
conflicting_entries = housing_data.groupby('code')['area'].nunique()
conflicts = conflicting_entries[conflicting_entries > 1]
if len(conflicts) > 0:
    for code in conflicts.index:
        unique_areas = housing_data[housing_data['code'] == code]['area'].unique()
        print(f"Conflicting areas for code {code}: {unique_areas}")
        # Assuming you want to choose the first area encountered for each code
        chosen_area = unique_areas[0]
        housing_data.loc[housing_data['code'] == code, 'area'] = chosen_area

# Verify that the conflicts have been resolved
conflicting_entries_after_fix = housing_data.groupby('code')['area'].nunique()
conflicts_after_fix = conflicting_entries_after_fix[conflicting_entries_after_fix > 1]
if len(conflicts_after_fix) == 0:
    print("Conflicts resolved. Unique 'area' for each 'code' now.")


# In[40]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

# Assuming 'housing_data' contains your dataset
# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(housing_data[['houses_sold', 'average_price', 'area', 'borough_flag']])
# Assuming 'no_of_crimes' is the target variable
y = housing_data['no_of_crimes']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate models
models = [
    LinearRegression(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    SVR(),
    KNeighborsRegressor()
]

# Train and evaluate each model
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model: {model.__class__.__name__}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}\n")


# In[36]:


from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Assuming 'housing_data' contains your dataset
# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(housing_data[['houses_sold', 'average_price', 'area', 'borough_flag']])
# Assuming 'no_of_crimes' is the target variable
y = housing_data['no_of_crimes']

# Initialize the RandomForestRegressor with the best parameters from the previous evaluation
best_model = RandomForestRegressor()

# Train the model on the entire dataset
best_model.fit(X, y)

# Predicting 'no_of_crimes' using the trained model
predictions = best_model.predict(X)

# Adding predictions to the housing_data DataFrame
housing_data['predicted_crimes'] = predictions

# View the housing_data DataFrame with predicted crime rates
print(housing_data)


# In[37]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd

# Assuming 'housing_data' contains your dataset
X = pd.get_dummies(housing_data[['houses_sold', 'average_price', 'area', 'borough_flag']])
y = housing_data['no_of_crimes']

# Define the parameter grid to search through
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30]  # Maximum depth of the trees
}

# Initialize the RandomForestRegressor
rf = RandomForestRegressor()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

# Fit the grid search to the data
grid_search.fit(X, y)

# Get the best parameters and best estimator
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

print("Best Parameters:", best_params)
print("Best Estimator:", best_estimator)


# In[38]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

# Assuming 'housing_data' contains your dataset
# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(housing_data[['houses_sold', 'average_price', 'area', 'borough_flag']])
# Assuming 'no_of_crimes' is the target variable
y = housing_data['no_of_crimes']

# Splitting the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestRegressor with the best parameters from the previous evaluation
best_model = RandomForestRegressor(max_depth=10, n_estimators=100)

# Train the model on the training dataset
best_model.fit(X_train, y_train)

# Predicting 'no_of_crimes' using the trained model on the test set
predictions = best_model.predict(X_test)

# You can evaluate the model's performance using metrics like Mean Squared Error (MSE) or R-squared
# For example:
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")


# In[ ]:




