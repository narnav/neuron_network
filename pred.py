import tensorflow as tf
import numpy as np

# Load the model
model = tf.keras.models.load_model('model.h5')



# Prepare the input data
input_data = np.array([[0.5, 0.3, 0.2]])

# Make predictions
predictions = model.predict(input_data)

# Print the predictions
print(predictions)