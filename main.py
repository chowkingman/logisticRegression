import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the data from CSV files
appointment_status = pd.read_csv('appointment_status.csv')
engagement_score = pd.read_csv('engagement_score.csv')
income_expense = pd.read_csv('income_expense.csv')
plan_start_date = pd.read_csv('plan_start_date.csv')
success_rate = pd.read_csv('success_rate.csv')

# Merge the dataframes on 'mock_id'
merged = pd.merge(appointment_status, engagement_score, on='mock_id')
merged = pd.merge(merged, income_expense, on='mock_id')
merged = pd.merge(merged, plan_start_date, on='mock_id')
merged = pd.merge(merged, success_rate, on='mock_id')

# Select the features we're interested in
features = merged[['Engagement Score', 'success_rate']]

# Handle missing values - this is a simple approach for demonstration purposes
# You might want to handle missing values in a way that makes sense for your data
features = features.dropna()

# Create the KMeans object and fit it to the data
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(features)

# Add the cluster labels to your DataFrame
features['cluster'] = kmeans.labels_

# Plot the clusters
plt.scatter(features['Engagement Score'], features['success_rate'], c=features['cluster'])
plt.xlabel('Engagement Score')
plt.ylabel('Success Rate')
plt.show()