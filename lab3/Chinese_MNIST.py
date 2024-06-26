# Import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Paths to CSV file and image folder
csv_path = "./Chinese_MNIST_Dataset/chinese_mnist.csv"
image_folder = "./Chinese_MNIST_Dataset/data/data"

# Load the CSV file containing metadata about the images
data_info = pd.read_csv(csv_path)

# Load an image based on the row information
def load_image(row):
    filename = f"input_{row['suite_id']}_{row['sample_id']}_{row['code']}.jpg"
    filepath = os.path.join(image_folder, filename)
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return image

# Load all images and corresponding labels
images = data_info.apply(load_image, axis=1).values
labels = data_info['code'].values

# Reshape images to be compatible with the classifiers
images = np.array([img.reshape(-1) for img in images])

# Evaluate classifier performance
def evaluate_classifier(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    return acc, prec, rec, f1, conf_matrix

# Plot confusion matrix using seaborn
def plot_confusion_matrix(conf_matrix, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(labels), yticklabels=np.unique(labels))
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Run experiments
def run_experiment(train_size):
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, test_size=1000, random_state=42)
    train_idx, test_idx = next(sss.split(images, labels))

    X_train, X_test = images[train_idx], images[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    classifiers = {
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Decision Tree": DecisionTreeClassifier(),
        "SGD": SGDClassifier(max_iter=250)
    }

    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        results[name] = evaluate_classifier(clf, X_test, y_test)
    
    return results

# Define training sizes for three stages
training_sizes = [5000, 10000, 14000]

# Run experiments for each stage
for size in training_sizes:
    print(f"Running experiment with training size: {size}")
    results = run_experiment(size)
    for clf_name, metrics in results.items():
        acc, prec, rec, f1, conf_matrix = metrics
        print(f"\nClassifier: {clf_name}")
        print(f"Accuracy: {acc}")
        print(f"Precision: {prec}")
        print(f"Recall: {rec}")
        print(f"F1 Score: {f1}\n")
        plot_confusion_matrix(conf_matrix, f"{clf_name} Confusion Matrix (Training Size: {size})")
