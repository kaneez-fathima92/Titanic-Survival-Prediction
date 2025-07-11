# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Step 2: Load the Dataset
df = pd.read_csv('titanic.csv.zip')  # Update path if necessary

# Step 3: Preprocessing
# Drop irrelevant columns
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Handle missing values properly
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare'] = df['Fare'].fillna(df['Fare'].median())  # Add this line

# Convert categorical features
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Check for any remaining NaNs
print("Missing values:\n", df.isnull().sum())


# Step 4: Feature and Target Selection
X = df.drop('Survived', axis=1)
y = df['Survived']

# Optional: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 6: Model Training
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 7: Prediction and Evaluation
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
