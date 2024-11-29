import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Generate 600 random points centered at C1 = (0, 0) with variance 10
np.random.seed(42)  # Set seed for reproducibility
mean1 = [0, 0]  # Center at (0, 0)
cov = [[10, 0], [0, 10]]  # Variance in x and y
points1 = np.random.multivariate_normal(mean1, cov, 600)

# Calculate distance from C1 and assign labels
distances1 = np.sqrt(points1[:, 0]**2 + points1[:, 1]**2)
labels1 = np.where(distances1 < 6, 0, 1)

# Scatter plot for Step 1
plt.figure(figsize=(8, 6))
plt.scatter(points1[labels1 == 0, 0], points1[labels1 == 0, 1], c='blue', label='Y=0 (dist < 6)', alpha=0.7)
plt.scatter(points1[labels1 == 1, 0], points1[labels1 == 1, 1], c='red', label='Y=1 (dist >= 6)', alpha=0.7)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.title('Scatter Plot of Points Centered at C1=(0, 0)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True)
plt.show()

# Step 2: Generate another random points centered at C2 = (10, 10) with variance 10
mean2 = [10, 10]  # Center at (10, 10)
points2 = np.random.multivariate_normal(mean2, cov, 600)

# Calculate distance from C2 and assign labels
distances2 = np.sqrt((points2[:, 0] - 10)**2 + (points2[:, 1] - 10)**2)
labels2 = np.where(distances2 < 3, 0, 1)

# Scatter plot for Step 2
plt.figure(figsize=(8, 6))
plt.scatter(points2[labels2 == 0, 0], points2[labels2 == 0, 1], c='blue', label='Y=0 (dist < 3)', alpha=0.7)
plt.scatter(points2[labels2 == 1, 0], points2[labels2 == 1, 1], c='red', label='Y=1 (dist >= 3)', alpha=0.7)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.title('Scatter Plot of Points Centered at C2=(10, 10)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True)
plt.show()

# Combine the points and labels
points = np.vstack((points1, points2))
labels = np.hstack((labels1, labels2))

# Step 3: Define x3 as a Gaussian function f(x1, x2)
def gaussian(x1, x2):
    return np.exp(-(x1**2 + x2**2) / 50)  # Example Gaussian function

x1, x2 = points[:, 0], points[:, 1]
x3 = gaussian(x1, x2)

# 3D Scatter plot for Step 3
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x1, x2, x3, c=labels, cmap='coolwarm', alpha=0.8)
ax.set_title('3D Scatter Plot of (X1, X2, X3) with Y Color')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
plt.colorbar(scatter, label='Y')
plt.show()

# Step 4: Add a linear hyperplane to separate Y=0 and Y=1
# Define a simple hyperplane z = a*x + b*y + c
xx, yy = np.meshgrid(np.linspace(-15, 25, 50), np.linspace(-15, 25, 50))
zz = -0.03 * xx - 0.03 * yy + 0.6  # Example hyperplane equation

# 3D Scatter plot with hyperplane
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x1, x2, x3, c=labels, cmap='coolwarm', alpha=0.8, label='Data Points')
ax.plot_surface(xx, yy, zz, color='lightblue', alpha=0.5)
ax.set_title('3D Scatter Plot with Linear Hyperplane')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
plt.show()
