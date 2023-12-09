import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

train_df = pd.read_csv('training.csv')
test_df = pd.read_csv('testing.csv')
reference_df = pd.read_csv('reference.csv')

X_train = train_df.drop(columns=[f'f{i}' for i in range(1, 13)])
y_train = train_df[[f'f{i}' for i in range(1, 13)]]
X_test = test_df.drop(columns=[f'f{i}' for i in range(1, 13)])
y_test = reference_df[[f'f{i}' for i in range(1, 13)]]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train_scaled, y_train)

predictions = knn.predict(X_test_scaled)

with open('knn_results.txt', 'w') as file:
    file.write("KNN Classification Report for each month:\n")
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

print("KNN results written to knn_results.txt")
