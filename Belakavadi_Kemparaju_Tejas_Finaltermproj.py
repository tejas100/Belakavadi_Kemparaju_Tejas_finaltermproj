#!/usr/bin/env python
# coding: utf-8

# CS634101 FINAL TERM PROJECT
# 
# Tejas Belakavadi Kemparaju - tb389
# 

#  

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import time
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
from prettytable import PrettyTable

# Import dataset
dataframe = pd.read_csv('eTransactionData.csv')

# Preprocessing 
features = dataframe.drop(columns=['Unnamed: 0', 'Transaction ID', 'Customer ID', 'Is Fraudulent', 'IP Address', 'Shipping Address', 'Billing Address'])
labels = dataframe['Is Fraudulent']

features['Transaction Date'] = pd.to_datetime(features['Transaction Date'])
features['Day'] = features['Transaction Date'].dt.day
features['Month'] = features['Transaction Date'].dt.month
features['Year'] = features['Transaction Date'].dt.year
features = features.drop(columns=['Transaction Date'])

encoder = LabelEncoder()
categorical_columns = ['Payment Method', 'Product Category', 'Customer Location', 'Device Used']
for column in categorical_columns:
    features[column] = encoder.fit_transform(features[column].astype(str))

# Handle missing values
value_imputer = SimpleImputer(strategy='mean')
features = pd.DataFrame(value_imputer.fit_transform(features), columns=features.columns)

# Normalize numerical columns
normalizer = StandardScaler()
features = pd.DataFrame(normalizer.fit_transform(features), columns=features.columns)

# Initialize KFold
fold_count = 10
kfold_splitter = KFold(n_splits=fold_count, shuffle=True, random_state=42)

# Defining models
algorithms = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'BiLSTM': Sequential([
        Bidirectional(LSTM(64, input_shape=(features.shape[1], 1))),
        Dense(1, activation='sigmoid')
    ])
}

performance_results = {algo_name: [] for algo_name in algorithms.keys()}
confusion_matrices = {algo_name: np.zeros((2, 2)) for algo_name in algorithms.keys()}

for algo_name, model in algorithms.items():
    metrics_per_fold = []
    total_runtime = 0
    
    for fold_idx, (train_idx, test_idx) in enumerate(kfold_splitter.split(features), 1):
        # Split data
        train_features, test_features = features.iloc[train_idx], features.iloc[test_idx]
        train_labels, test_labels = labels.iloc[train_idx], labels.iloc[test_idx]

        if algo_name == 'BiLSTM':
            train_features = np.expand_dims(train_features.values, axis=2)
            test_features = np.expand_dims(test_features.values, axis=2)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        start_time = time.time()
        
        if algo_name == 'BiLSTM':
            model.fit(train_features, train_labels, epochs=10, batch_size=32, verbose=0)
            predicted_probabilities = model.predict(test_features).ravel()
        else:
            model.fit(train_features, train_labels)
            predicted_probabilities = model.predict_proba(test_features)[:, 1]
        
        runtime = time.time() - start_time
        total_runtime += runtime
        
        # Binarize predictions for threshold 0.5
        predicted_labels = (predicted_probabilities >= 0.5).astype(int)
        
        # Calculate confusion matrix
        fold_cm = confusion_matrix(test_labels, predicted_labels)
        confusion_matrices[algo_name] += fold_cm
        
        tn, fp, fn, tp = fold_cm.ravel()
        
        true_positive_rate = tp / (tp + fn)
        false_positive_rate = fp / (fp + tn)
        true_negative_rate = tn / (tn + fp)
        false_negative_rate = fn / (fn + tp)
        
        precision = precision_score(test_labels, predicted_labels)
        accuracy = accuracy_score(test_labels, predicted_labels)
        recall = recall_score(test_labels, predicted_labels)
        error_rate = 1 - accuracy
        f1 = f1_score(test_labels, predicted_labels)
        
        balanced_accuracy = (true_positive_rate + true_negative_rate) / 2
        true_skill_statistic = true_positive_rate - false_positive_rate
        heidke_skill_score = (2 * (tp * tn - fp * fn)) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
        
        brier_score = np.mean((predicted_probabilities - test_labels) ** 2)
        brier_skill_score = 1 - brier_score / np.var(test_labels)
        
        roc_auc = roc_auc_score(test_labels, predicted_probabilities)
        
        metrics_per_fold.append({
            'Fold': f'Fold {fold_idx}',
            'TPR': true_positive_rate,
            'FPR': false_positive_rate,
            'TNR': true_negative_rate,
            'FNR': false_negative_rate,
            'Precision': precision,
            'Accuracy': accuracy,
            'Recall': recall,
            'Error Rate': error_rate,
            'F1 Score': f1,
            'BACC': balanced_accuracy,
            'TSS': true_skill_statistic,
            'HSS': heidke_skill_score,
            'BS': brier_score,
            'BSS': brier_skill_score,
            'ROC_AUC': roc_auc,
            'Runtime (seconds)': runtime
        })
    
    # Add average metrics
    avg_metrics = {metric: np.mean([fold[metric] for fold in metrics_per_fold if metric != 'Fold']) for metric in metrics_per_fold[0] if metric != 'Fold'}
    avg_metrics['Fold'] = 'Average'
    avg_metrics['Total Runtime'] = total_runtime
    
    metrics_per_fold.append(avg_metrics)
    performance_results[algo_name] = metrics_per_fold

# Print metrics in a table
def display_table(data, header, confusion_matrix=None):
    dataframe = pd.DataFrame(data)

    essential_metrics = ['TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Accuracy', 'Recall', 'Error Rate', 
                        'F1 Score', 'BACC', 'TSS', 'HSS', 'BS', 'BSS', 'ROC_AUC', 'Runtime (seconds)']
    for metric in essential_metrics:
        if metric not in dataframe.columns:
            dataframe[metric] = np.nan
    
    if 'Total Runtime' in dataframe.iloc[-1]:
        dataframe.loc[dataframe.index[-1], 'Runtime (seconds)'] = dataframe.iloc[-1]['Total Runtime']
    
    selected_dataframe = dataframe[['Fold'] + essential_metrics]
    selected_dataframe = selected_dataframe.iloc[:7]  # Display only the first 7 folds
    transposed_df = selected_dataframe.set_index('Fold').transpose()
    
    table = PrettyTable()
    table.title = header
    table.field_names = ['Metric'] + list(transposed_df.columns)
    table.align = 'r'
    table.float_format = '.4'
    
    for index, row in transposed_df.iterrows():
        table.add_row([index] + list(row))

    if confusion_matrix is not None:
        # Add confusion matrix values as additional rows
        cm_values = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
        cm_flattened = confusion_matrix.ravel()
        for label, value in zip(cm_values, cm_flattened):
            table.add_row([label] + ['-' for _ in range(len(transposed_df.columns) - 1)] + [value])

    print(table)

# Print tables for each model
for algo_name, metrics in performance_results.items():
    print(f"\n{algo_name} Results:")
    display_table(metrics, f"{algo_name} Metrics:", confusion_matrix=confusion_matrices[algo_name])
    print("\n" + "="*50 + "\n")

# Identify and print the fastest algorithm based on total runtime
fastest_algo = min(performance_results.keys(), key=lambda k: performance_results[k][-1]['Total Runtime'])


# In[ ]:




