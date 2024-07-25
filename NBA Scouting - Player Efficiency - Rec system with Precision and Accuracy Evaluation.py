#!/usr/bin/env python
# coding: utf-8

# ## Load CSV File

# In[248]:


#Check out efficiency by player, then team individually. 
#Work on a combining dataset to find efficient players by selecting a team later
##dataframe.iloc[:, :17].head()
import pandas as pd
import numpy as np

# Load the CSV file
file_path = 'Advanced.csv'
advanced_df = pd.read_csv(file_path)

# # Display the first few rows to understand the structure
# print("Advanced Stats Data:")
advanced_df.iloc[:, 10:].head()


# ## Preprocess Data

# In[249]:


# Filter for the 2024 season
advanced_df = advanced_df[advanced_df['season'] == 2024]

# Handling missing values
advanced_df = advanced_df.dropna(subset=['player', 'tm', 'mp', 'per'])

# Normalize numerical values (PER and USG%)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
advanced_df[['mp', 'per']] = scaler.fit_transform(advanced_df[['mp', 'per']])

# # Display the preprocessed dataframe
# print("\nPreprocessed Data:")
# print(advanced_df.head())


# ## Prep Dataset

# In[250]:


from sklearn.model_selection import train_test_split

# Define features (PER and USG%)
features = advanced_df[['mp', 'per']]

# Split the data into training and test sets
X_train, X_test = train_test_split(features, test_size=0.2, random_state=42)

# # Display the shapes of the train and test sets
# print("\nTraining set shape:", X_train.shape)
# print("Test set shape:", X_test.shape)


# ## Build and Train Model 

# In[251]:


from sklearn.cluster import KMeans

# Build the KMeans model
kmeans = KMeans(n_clusters=3, random_state=42) 

# Train the model
kmeans.fit(X_train)

# Predict the clusters for the test set
test_clusters = kmeans.predict(X_test)

# Add cluster labels to the original dataframe for interpretation
advanced_df['Cluster'] = kmeans.predict(features)

# # Display the dataframe with cluster labels
print("\nData with Cluster Labels:")
print(advanced_df.head())


# ## Identify Efficency of Players

# In[252]:


# Sort players within each cluster based on PER and USG%
top_players = advanced_df.sort_values(by=['Cluster', 'mp', 'per'], ascending=[True, False, False])

def get_best_players_for_team(team_name, data, top_n = 3):
    # Filter the data for the given team
    team_data = data[data['tm'] == team_name]
 
    # Check if the team data is empty
    if team_data.empty:
        return f"No data available for team {team_name}"
 
    # Sort the team data by PER and USG%
    sorted_team_data = team_data.sort_values(by=['mp', 'per'], ascending=[False, False])
   
    # Select the top N players
    best_players = sorted_team_data.head(top_n)
 
    return best_players

# Example usage
team_name = 'NYK'  # Replace with the desired team name
best_players = get_best_players_for_team(team_name, advanced_df)
print(f"\nBest players for {team_name}:\n{best_players.iloc[:, :4]}")


# ## Evaluate Model 

# In[253]:


from sklearn.metrics import silhouette_score

# Calculate the silhouette score for the training data
silhouette_avg = silhouette_score(X_train, kmeans.labels_)
print("Silhouette Score for Training Data:", silhouette_avg)

# Calculate the silhouette score for the test data
silhouette_avg_test = silhouette_score(X_test, test_clusters)
print("Silhouette Score for Test Data:", silhouette_avg_test)

# Inertia
# Calculate the inertia for the training data
inertia_train = kmeans.inertia_
print("Inertia for Training Data:", inertia_train)

# For test data, we can use the predict method and calculate inertia manually
test_distances = kmeans.transform(X_test)
inertia_test = sum(np.min(test_distances, axis=1)**2)
print("Inertia for Test Data:", inertia_test)

#Elbow Method
import matplotlib.pyplot as plt

# Use the elbow method to determine the optimal number of clusters
inertia = []
for n in range(1, 11):
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(features)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

