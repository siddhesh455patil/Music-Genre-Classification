import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the extracted features
df = pd.read_csv("features/genre_features.csv")

# Separate features and labels
X = df.drop(columns=["label"])
y = df["label"]

# Encode genre labels as numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save the encoder for later use in prediction
os.makedirs("models", exist_ok=True)
joblib.dump(label_encoder, "models/label_encoder.pkl")

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, "models/scaler.pkl")

# Train/test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Output data to verify
print("Preprocessing complete")
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Testing labels shape:", y_test.shape)
print("Data split into training and testing sets")

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("\n Random Forest Results")
print(classification_report(y_test, rf_pred))
print("Accuracy:", accuracy_score(y_test, rf_pred))

# Save model
joblib.dump(rf, "models/random_forest.pkl")

# Train SVM
svm = SVC(kernel="rbf", C=10, gamma=0.01)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print("\n SVM Results")
print(classification_report(y_test, svm_pred))
print("Accuracy:", accuracy_score(y_test, svm_pred))
joblib.dump(svm, "models/svm_model.pkl")

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print("\n KNN Results")
print(classification_report(y_test, knn_pred))
print("Accuracy:", accuracy_score(y_test, knn_pred))
joblib.dump(knn, "models/knn_model.pkl")

# Plot confusion matrix for best model (Random Forest shown here)
plt.figure(figsize=(10, 6))
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()