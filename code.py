import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter

class CustomKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))

    def cosine_distance(self, x1, x2):
        return 1 - cosine_similarity([x1], [x2])[0, 0]

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        manhattan_distances = [self.manhattan_distance(x, x_train) for x_train in self.X_train]
        cosine_distances = [self.cosine_distance(x, x_train) for x_train in self.X_train]

        combined_distances = [0.5 * manhattan + 0.5 * cosine for manhattan, cosine in zip(manhattan_distances, cosine_distances)]

        k_indices = np.argsort(combined_distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Load the Iris dataset from sklearn
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the custom KNN classifier
custom_knn = CustomKNN(k=3)
custom_knn.fit(X_train, y_train)

# Predict the classes for the test data
y_pred = custom_knn.predict(X_test)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)
