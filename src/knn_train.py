import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from knn_utils import plot_confusion_matrix
import os

# 1. Load dataset

# Automatically detect CSV path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(project_root, 'data', 'iris.dataset.csv')

# Load dataset
df = pd.read_csv(csv_path)
print("CSV loaded successfully!")
print(df.head())


# 2. Split features and target
X = df.drop('species', axis=1)  # change 'species' if your target column name differs
y = df['species']

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train KNN for multiple K values
k_values = [1, 3, 5, 7, 9, 11]
accuracy_list = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    accuracy_list.append(acc)
    
    print(f"\n--- K = {k} ---")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    plot_confusion_matrix(y_test, y_pred, classes=knn.classes_, title=f"Confusion Matrix (K={k})")

# 6. Plot accuracy vs K
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.plot(k_values, accuracy_list, marker='o')
plt.title("KNN: Accuracy vs K")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.show()
