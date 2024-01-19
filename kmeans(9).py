import open3d as o3d
import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import cm

class KMeans:
    def __init__(self, data, normals, colors, n_clusters):
        self.data = data
        self.colors = colors
        self.normals = normals
        self.n_clusters = n_clusters
        self.normalized_matrix = None
        self.centroids = self.centroid_init()
        self.labels = np.zeros(len(self.data), dtype=int)

    def normalization(self):
        matrix_points = np.asarray(self.data)
        matrix_normals = np.asarray(self.normals)
        matrix_colors = np.asarray(self.colors)
        combined_matrix = np.hstack((matrix_points, matrix_normals, matrix_colors))
        self.normalized_matrix = (combined_matrix - combined_matrix.min(axis=0)) / (combined_matrix.max(axis=0) - combined_matrix.min(axis=0))

    def centroid_init(self):
        self.normalization()
        centroids_indices = np.random.choice(self.normalized_matrix.shape[0], self.n_clusters, replace=False)
        self.centroids = self.normalized_matrix[centroids_indices]
        self.centroids = self.centroids.astype(self.normalized_matrix.dtype)
        return self.centroids

    def update_centroids(self):
        new_centroids = []
        for cluster_index in range(self.n_clusters):
            cluster_points = self.normalized_matrix[self.labels == cluster_index]
            if len(cluster_points) > 0:
                new_centroid = np.mean(cluster_points, axis=0)
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(self.centroids[cluster_index])
        self.centroids = new_centroids

    def fit(self, max_iterations=1000000000):
        for _ in range(max_iterations):
            old_centroids = self.centroids
            distances = cdist(self.normalized_matrix, self.centroids, 'euclidean')
            self.labels = np.argmin(distances, axis=1)
            self.update_centroids()
            new_centroids = np.array(self.centroids)
            if np.allclose(old_centroids, new_centroids, atol=1e-8):
                break

    def visualize_clusters(self):
        viridis_colors = cm.tab20(np.linspace(0, 1, max(self.labels) + 1))
        cluster_colors = [viridis_colors[i] for i in self.labels]

        # 转换为Vector3dVector类型
        cluster_colors = np.asarray(cluster_colors)
        cluster_colors = cluster_colors[:, :3]  # 仅保留RGB部分
        cluster_colors = o3d.utility.Vector3dVector(cluster_colors)

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.asarray(self.data))
        point_cloud.colors = cluster_colors

        o3d.visualization.draw_geometries([point_cloud])
    # 使用示例
pcd = o3d.io.read_point_cloud("fragment.ply")
kmeans = KMeans(pcd.points,pcd.normals,pcd.colors, n_clusters=20)
kmeans.normalization()
kmeans.centroid_init()
kmeans.fit()
kmeans.visualize_clusters()











