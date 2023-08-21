import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load and preprocess your data
# ...

# Split the data into training, validation, and test sets
# ...

# Create a neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(input_shape,)),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=epochs, validation_data=(val_data, val_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data, test_labels)
print("Test accuracy:", test_acc)

# Save the model to a file
model.save('model.h5')
