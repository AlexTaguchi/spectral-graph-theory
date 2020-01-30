# Modules
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.spatial import distance_matrix

# Build a uniform distribution of nodes as a 10x10 grid
coordinates = [(x // 10, x % 10) for x in range(100)]

# Construct normalized distance Laplacian
distances = distance_matrix(coordinates, coordinates)
inverse_distances = np.divide(1, distances, where=distances!=0)
degrees = np.sum(inverse_distances, axis=0)
identity = np.eye(distances.shape[0])
laplacian = identity - inverse_distances / np.sqrt(np.outer(degrees, degrees))

# Calculate the eigenvalues and eigenvectors of the Laplacian
eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

# Compare 10x10 grid with second and third eigenvectors embeddings
colors = [(x/100, 0.5, 0.5) for x in range(100)]
fig = plt.figure()
ax = fig.add_subplot(121)
ax.set_aspect('equal')
ax.scatter([x // 10 for x in range(100)], [x % 10 for x in range(100)], color=colors)
ax = fig.add_subplot(122)
ax.set_aspect('equal')
ax.scatter(eigenvectors[:, 1], eigenvectors[:, 2], color=colors)
plt.show()