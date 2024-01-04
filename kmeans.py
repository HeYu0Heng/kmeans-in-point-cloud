import open3d as o3d
import numpy as np
import random

class KMeans:
    def __init__(self, data, n_clusters):
        self.data = data
        self.n_clusters = n_clusters
        self.centroids = self.centroid_init()
        self.labels = np.zeros(len(self.data), dtype=int)


    def centroid_init(self):
        data_min = np.min(self.data, axis=0)
        data_max = np.max(self.data, axis=0)
        centroids = []
        for _ in range(self.n_clusters):
            centroid = [random.uniform(min_val, max_val) for min_val, max_val in zip(data_min, data_max)]
            centroids.append(centroid)
        return centroids

    def update_centroids(self):
        new_centroids = []
        for cluster_index in range(self.n_clusters):
            cluster_points = self.data[self.labels == cluster_index]
            if len(cluster_points) > 0:
                new_centroid = np.mean(cluster_points, axis=0)
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(self.centroids[cluster_index])
        self.centroids = new_centroids


    def fit(self,max_iterations = 1000000000):
        for _ in range(max_iterations):
            old_centroids = np.array(self.centroids)
            distances = np.linalg.norm(self.data[:, None] - old_centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)
            self.update_centroids()
            new_centroids = np.array(self.centroids)
            if np.allclose(old_centroids, new_centroids, atol=1e-8):
                break


    def visualize_clusters(self):
        colors = np.random.rand(self.n_clusters, 3)
        pcd_colors = colors[self.labels]

        point_clouds = []
        for i in range(self.n_clusters):
            cluster_points = self.data[self.labels == i]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cluster_points)
            pcd.colors = o3d.utility.Vector3dVector(pcd_colors[self.labels == i])
            point_clouds.append(pcd)

        o3d.visualization.draw_geometries(point_clouds)

    # 使用示例
pcd = o3d.io.read_point_cloud("bunny.ply")
pcd_np = np.asarray(pcd.points)
kmeans = KMeans(pcd_np, n_clusters=3)
kmeans.fit()
kmeans.visualize_clusters()










