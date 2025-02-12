import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('../data/forestfires_extended.csv')

# Data Preprocessing (assuming month and day are categorical variables)
data['month'] = data['month'].astype('category').cat.codes
data['day'] = data['day'].astype('category').cat.codes

# Define significant fire: if area > 0, we consider it as fire
data['fire'] = np.where(data['area'] > 0, 1, 0)

# Select features and target
features = ['temp', 'RH', 'wind']
target = 'fire'

X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(probability=True),  # Set probability=True for predict_proba
    'Gradient Boosting': GradientBoostingClassifier()
}

# Dictionary to store accuracy results
results = {}

# Train and evaluate each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy
    print(f'{model_name} Accuracy: {accuracy:.2f}')
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Fire', 'Fire'], yticklabels=['No Fire', 'Fire'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()

# Find the model with the highest accuracy
best_model_name = max(results, key=results.get)
best_model_accuracy = results[best_model_name]
best_model = models[best_model_name]

print(f'Best Model: {best_model_name} with Accuracy: {best_model_accuracy:.2f}')

# Save the best model
with open('./best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("Best model saved successfully.")