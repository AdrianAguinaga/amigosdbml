# -*- coding: utf-8 -*-
"""
Create by AdrianRA
Low complexity neural network to analyze EEG, ECG and Galvanic responses
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import tensorflow as tf

tf.config.list_physical_devices('GPU')

# Load the data
df = pd.read_csv('Path_of_the_csv_file')

# Replace NaN values with the mean of the column
df.fillna(df.mean(), inplace=True)

# Separate features and labels
X = df.drop(columns=['Class'])
y = df['Class']

# Normalize the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Define KFold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store metrics for each fold
accuracy_list, recall_list, precision_list, f1_list = [], [], [], []

fold_no = 1

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Create the model with adjusted regularization
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.0001)),
        Dropout(0.2),
        Dense(32, activation='relu', kernel_regularizer=l2(0.0001)),
        Dropout(0.2),
        Dense(4, activation='softmax')
    ])

    # Compile the model with Nadam optimizer
    optimizer = Nadam()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model with the best parameters
    history = model.fit(X_train, y_train, epochs=100, batch_size=30, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

    # Get the predictions of the model
    y_pred = np.argmax(model.predict(X_test), axis=-1)

    # Calculate statistics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # Store the metrics
    accuracy_list.append(accuracy)
    recall_list.append(recall)
    precision_list.append(precision)
    f1_list.append(f1)

    print(f"Fold {fold_no} - Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1-Score: {f1:.4f}")
    fold_no += 1

# Calculate and print average metrics
avg_accuracy = np.mean(accuracy_list)
avg_recall = np.mean(recall_list)
avg_precision = np.mean(precision_list)
avg_f1 = np.mean(f1_list)

print(f"Average Accuracy: {avg_accuracy:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average F1-Score: {avg_f1:.4f}")

# Save average statistics to a CSV
stats_df = pd.DataFrame({
    "Metric": ["Average Accuracy", "Average Recall", "Average Precision", "Average F1-Score"],
    "Value": [avg_accuracy, avg_recall, avg_precision, avg_f1]
})
stats_df.to_csv('cross_validation_statistics.csv', index=False)

# Optionally, plot the training and validation loss over epochs for the last fold
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')
plt.show()

# Get the predictions of the model for the last fold
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot and save confusion matrix for the last fold
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()
