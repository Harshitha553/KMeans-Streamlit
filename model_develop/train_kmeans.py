import pandas as pd
import pickle

from sklearn.cluster import KMeans

from backend.preprocess import preprocess_data

df = pd.read_csv("datasets/Mall_Customers.csv")

X_scaled = preprocess_data(df)

kmeans = KMeans(
    n_clusters=5,
    random_state=42
)

kmeans.fit(X_scaled)

with open("kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

print("Model Saved Successfully")