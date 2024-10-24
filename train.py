import tensorflow as tf
from models.tcmh import TCMH
from utils.data_loader import data_generator
import config

# Define the model
B = config.BATCH_SIZE  # Batch size
input_shape = (500, 1)
model = TCMH(input_shape=input_shape)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.build([tf.TensorShape((None, 500, 1)) for _ in range(10)])
model.summary()

# Create Dataset
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(config.train_file_paths, B),
    output_signature=(
        [tf.TensorSpec(shape=(None, 500, 1), dtype=tf.float32) for _ in range(10)],
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )
).prefetch(tf.data.experimental.AUTOTUNE)

# Training the model
model.fit(train_dataset, epochs=config.EPOCHS, steps_per_epoch=len(config.train_file_paths) // B)

# Evaluation of the model
test_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(config.test_file_paths, B),
    output_signature=(
        [tf.TensorSpec(shape=(None, 500, 1), dtype=tf.float32) for _ in range(10)],
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )
).prefetch(tf.data.experimental.AUTOTUNE)

loss, accuracy = model.evaluate(test_dataset, steps=len(config.test_file_paths) // B)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Make predictions
predictions = model.predict(test_dataset, steps=len(config.test_file_paths) // B)
print('Predictions:', predictions)
