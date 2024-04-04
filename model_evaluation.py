import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# Assuming y_true are the true labels and y_pred are the predicted labels from your model
# Replace y_true and y_pred with your actual data
y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1])  # Example true labels
y_pred = np.array([0, 1, 1, 1, 1, 0, 1, 0, 1, 1])  # Example predicted labels

# Calculate classification metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=['Predicted 0', 'Predicted 1'], 
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Generate classification report
class_names = ['Class 0', 'Class 1']  # Assuming binary classification
report = classification_report(y_true, y_pred, target_names=class_names)
print("Classification Report:\n", report)
