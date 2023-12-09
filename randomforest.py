import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('training.csv')
df_test = pd.read_csv('testing.csv')
df_reference = pd.read_csv('reference.csv')

X_train = df.drop(columns=[f'f{i}' for i in range(1, 13)])
y_train = df[[f'f{i}' for i in range(1, 13)]]
X_test = df_test.drop(columns=[f'f{i}' for i in range(1, 13)])
y_test = df_reference[[f'f{i}' for i in range(1, 13)]]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

predictions = rf_classifier.predict(X_test_scaled)

with open('randomforest_results.txt', 'w') as file:
    file.write("Classification Report for each month:\n")
    overall_accuracy = 0
    for i in range(1, 13):
        report = classification_report(y_test[f'f{i}'], predictions[:, i-1], zero_division=0)
        file.write(f"\nMonth {i} Report:\n")
        file.write(report)
        accuracy = accuracy_score(y_test[f'f{i}'], predictions[:, i-1])
        overall_accuracy += accuracy
        file.write(f"Accuracy for Month {i}: {accuracy}\n")

    overall_accuracy /= 12
    file.write(f"\nAverage Accuracy Across All Months: {overall_accuracy}\n")

print("Results written to randomforest_results.txt")
