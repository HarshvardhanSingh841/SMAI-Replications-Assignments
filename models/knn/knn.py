import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = self._compute_distances(x)
        # Get the k nearest labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Convert one-hot encoded labels to class indices
        k_nearest_labels = [np.argmax(label) for label in k_nearest_labels]
        
        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def _compute_distances(self, x):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(self.X_train - x), axis=1)
        else:
            raise ValueError("Unsupported distance metric")

    def evaluate(self, X_val, y_val):
        y_pred = self.predict(X_val)
        
        # Convert one-hot encoded y_val to class indices
        #y_val_indices = np.argmax(y_val, axis=1)

        if y_val.ndim > 1:
            y_val_indices = np.argmax(y_val, axis=1)
        else:
            y_val_indices = y_val
        
        accuracy = self._accuracy(y_val_indices, y_pred)
        precision = self._precision(y_val_indices, y_pred)
        recall = self._recall(y_val_indices, y_pred)
        f1 = self._f1_score(y_val_indices, y_pred)
        return accuracy, precision, recall, f1

    def _accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
    
    def _precision(self, y_true, y_pred):
        classes = np.unique(y_true)
        precision_scores = {}
        for cls in classes:
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fp = np.sum((y_pred == cls) & (y_true != cls))
            precision_scores[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0
        return np.mean(list(precision_scores.values()))
    
    def _recall(self, y_true, y_pred):
        classes = np.unique(y_true)
        recall_scores = {}
        for cls in classes:
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fn = np.sum((y_pred != cls) & (y_true == cls))
            recall_scores[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0
        return np.mean(list(recall_scores.values()))
    
    def _f1_score(self, y_true, y_pred):
        precision = self._precision(y_true, y_pred)
        recall = self._recall(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

