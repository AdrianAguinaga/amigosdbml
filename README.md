# amigosdbml
This is a contribution of a neural network develop and optimized to analize multimodal signals from the AMIGOS Dataset. 

Python program designed for analyzing electroencephalogram (EEG), electrocardiogram (ECG), and galvanic skin response data using a low-complexity neural network. It's structured to provide a comprehensive workflow for data preprocessing, model training, evaluation, and visualization. Here's a breakdown of its components:

A sample to test can be downloaded in: https://filebin.net/tef9uvidqajcip81

Import Libraries: It begins by importing necessary libraries such as numpy, pandas, sklearn, seaborn, matplotlib, and tensorflow. These libraries provide tools for data manipulation, machine learning model development, and visualization.

Data Loading and Preprocessing:

The data is loaded using pandas.
It replaces missing values with the mean of the respective columns.
The features (X) and labels (y) are separated, and the data is split into training and testing sets.
The data is normalized using MinMaxScaler.
Neural Network Model Creation:

A Sequential model is defined using TensorFlow's Keras API.
The model consists of Dense layers with ReLU activation functions and a final layer with softmax activation suitable for multi-class classification.
The Nadam optimizer is used for compiling the model.
Training the Model:

The model is trained on the training dataset for a large number of epochs (50,000) with a batch size of 30.
Validation data is used to monitor the performance on unseen data during training.
Visualization:

Plots for training and validation loss and accuracy are generated and saved, providing visual feedback on the learning process.
Model Evaluation:

The model’s performance is evaluated on the test set.
Metrics such as accuracy, recall, precision, and F1 score are computed.
A confusion matrix is generated and visualized to understand the model's performance across different classes.
Saving Results:

The performance metrics are saved to a CSV file for further analysis or reporting.
