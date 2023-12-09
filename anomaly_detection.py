import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv('training.csv')
test_df = pd.read_csv('testing.csv')
reference_df = pd.read_csv('reference.csv')

X_train = train_df.drop(columns=[f'f{i}' for i in range(1, 13)])
X_test = test_df.drop(columns=[f'f{i}' for i in range(1, 13)])
y_reference = reference_df[[f'f{i}' for i in range(1, 13)]]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

iso_forest = IsolationForest(n_estimators=100, contamination=0.5, random_state=42)
iso_forest.fit(X_train_scaled)

anomalies = iso_forest.predict(X_test_scaled)

test_df['anomaly'] = anomalies

anomalous_data = test_df[test_df['anomaly'] == -1]

with open('a_d_results.txt', 'w') as file:
    file.write(f"Total data points in testing set: {len(test_df)}\n")
    file.write(f"Number of anomalies detected: {len(anomalous_data)}\n")

    file.write("\nAnomaly Detection Comparison with Reference Data:\n")
    for i in range(1, 13):
        anomaly_count = anomalous_data[anomalous_data[f'f{i}'] == 1].shape[0]
        actual_fire_count = y_reference[y_reference[f'f{i}'] == 1].shape[0]
        file.write(f"Month {i} - Detected Anomalies: {anomaly_count}, Actual Fires: {actual_fire_count}\n")

print("Anomaly detection results written to a_d_results.txt")
