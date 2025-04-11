import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from tensorflow.keras.models import Sequential 
from tensorflow.keras.models import Sequential
model = Sequential()
from tensorflow.keras.layers import LSTM, Dense, Dropout  
import joblib  

# Load NSL-KDD dataset (already preprocessed with only numerical features)  
df = pd.read_csv("NSL_KDD_binary.csv")  
X = df.drop("label", axis=1)  
y = df["label"]  # 0 = normal, 1 = attack  

# Scale features  
scaler = StandardScaler()  
X_scaled = scaler.fit_transform(X)  
X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])  

# Train/Test Split  
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)  

# LSTM Model  
model = Sequential()  
model.add(LSTM(64, input_shape=(1, X.shape[1]), return_sequences=False))  
model.add(Dropout(0.3))  
model.add(Dense(1, activation='sigmoid'))  

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1)  

# Save model and scaler  
model.save("lstm_model.h5")  
joblib.dump(scaler, "scaler.pkl")
