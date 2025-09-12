import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("fraud_dataset_sample.csv")

# Preprocessing
df.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)
df['type'] = LabelEncoder().fit_transform(df['type'])

# EDA Plots
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('heatmap.png')
plt.close()

sns.countplot(x='isFraud', data=df)
plt.title('Fraud Distribution')
plt.savefig('fraud_distribution.png')
plt.close()

sns.countplot(x='type', data=df)
plt.title('Transaction Types')
plt.savefig('transaction_type_distribution.png')
plt.close()

sns.boxplot(x='isFraud', y='amount', data=df)
plt.title('Amount vs Fraud')
plt.savefig('amount_by_fraud.png')
plt.close()

# Model Training
X = df.drop('isFraud', axis=1)
y = df['isFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()
