import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
# Generate some synthetic data (e.g., 2D points)
data = np.random.rand(1000, 10)  # 1000 samples, 10 features
print("Original Data Shape:", data.shape)
# Define the Autoencoder architecture
input_dim = data.shape[1]  # Input dimension (10 features)
# Encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(6, activation='relu')(input_layer)  # Compressed to 6 features
encoded = Dense(3, activation='relu')(encoded)      # Compressed to 3 features
# Decoder
decoded = Dense(6, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)  # Reconstruct original features
# Build Autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoded)
# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')
# Train the Autoencoder
autoencoder.fit(data, data, epochs=50, batch_size=32, shuffle=True, verbose=1)
# Use the encoder part for dimensionality reduction
encoder = Model(inputs=input_layer, outputs=encoded)
reduced_data = encoder.predict(data)
print("Reduced Data Shape:", reduced_data.shape)
