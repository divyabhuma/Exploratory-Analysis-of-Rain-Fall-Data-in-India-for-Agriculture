import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("weather.csv")

# Features and label
X = data.drop("RainTomorrow", axis=1)
y = data["RainTomorrow"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save files
pickle.dump(model, open("Rainfall.pkl", "wb"))
pickle.dump(scaler, open("scale.pkl", "wb"))

print("MODEL CREATED SUCCESSFULLY")
