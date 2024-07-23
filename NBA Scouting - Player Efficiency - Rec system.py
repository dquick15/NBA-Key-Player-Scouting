#!/usr/bin/env python
# coding: utf-8

# ## Load CSV File

##dataframe.iloc[:, :17].head()
import pandas as pd

# Load the CSV file
file_path = 'Advanced.csv'
advanced_df = pd.read_csv(file_path)

# # Display the first few rows to understand the structure
# print("Advanced Stats Data:")
# print(advanced_df.head())


# ## Preprocess Data


# Filter for the 2024 season
advanced_df = advanced_df[advanced_df['season'] == 2024]

# Handling missing values
advanced_df = advanced_df.dropna(subset=['player', 'tm', 'per', 'usg_percent'])

# Normalize numerical values (PER and USG%)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
advanced_df[['per', 'usg_percent']] = scaler.fit_transform(advanced_df[['per', 'usg_percent']])

# # Display the preprocessed dataframe
# print("\nPreprocessed Data:")
# print(advanced_df.head())


# ## Prep Dataset


from sklearn.model_selection import train_test_split

# Define features (PER and USG%)
features = advanced_df[['per', 'usg_percent']]

# Split the data into training and test sets
X_train, X_test = train_test_split(features, test_size=0.2, random_state=42)

# # Display the shapes of the train and test sets
# print("\nTraining set shape:", X_train.shape)
# print("Test set shape:", X_test.shape)


# ## Build and Train Model 


from sklearn.cluster import KMeans

# Build the KMeans model
kmeans = KMeans(n_clusters=5, random_state=42)  # Assuming we want to identify 5 clusters

# Train the model
kmeans.fit(X_train)

# Predict the clusters for the test set
test_clusters = kmeans.predict(X_test)

# Add cluster labels to the original dataframe for interpretation
advanced_df['Cluster'] = kmeans.predict(features)

# # Display the dataframe with cluster labels
# print("\nData with Cluster Labels:")
# print(advanced_df.head())


# ## Identify Efficency of Players


# Sort players within each cluster based on PER and USG%
top_players = advanced_df.sort_values(by=['Cluster', 'per', 'usg_percent'], ascending=[True, False, False])

def get_best_player_for_team(team_name, data):
    # Filter the data for the given team
    team_data = data[data['tm'] == team_name]
   
    # Check if the team data is empty
    if team_data.empty:
        return f"No data available for team {team_name}"
   
    # Find the player with the highest PER and USG%
    best_player = team_data.loc[team_data[['per', 'usg_percent']].mean(axis=1).idxmax()]
   
    return best_player

# Example usage
team_name = 'NYK'  # Replace with the desired team name
best_player = get_best_player_for_team(team_name, advanced_df)
# print(f"\nBest player for team {team_name}:\n{best_player}")

