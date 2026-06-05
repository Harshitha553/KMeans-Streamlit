from backend.clustering import predict_cluster

cluster = predict_cluster(
    income=50,
    score=60
)

print("Cluster:", cluster)