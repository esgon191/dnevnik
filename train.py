import logging.config
import tensorflow as tf
from models.TCMH import TCMH
from utils.data_loader import data_generator
import config
import logging

# Logging setup
logging.basicConfig(
    level=logging.DEBUG,  # Уровень логгирования
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Формат сообщения
    filename='logs/train.log',  # Имя файла для логов
    filemode='w'  # Режим записи (a - добавление, w - перезапись)
)

train_logger = logging.getLogger(name="train_logger")

# Define the model
B = config.BATCH_SIZE  # Batch size
input_shape = (500, 1)
model = TCMH(input_shape=input_shape)

train_logger.info('Created model')

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

train_logger.info('Compiled model')

# Summary of the model
model.build([tf.TensorShape((None, 500, 1)) for _ in range(10)])
model.summary()

train_logger.info('Model builded')

train_logger.info('Dataset creation')
# Create Dataset
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(config.X_train_file_path, config.y_train_file_path, B),
    output_signature=(
        tf.TensorSpec(shape=(10, None, 500, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )
).prefetch(tf.data.experimental.AUTOTUNE)

# Training the model
train_logger.info('Learning started')
model.fit(train_dataset, epochs=config.EPOCHS, steps_per_epoch=len(config.X_train_file_path) // B)

train_logger.info('Learning ended')

# Evaluation of the model
train_logger.info('Testing started')
test_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(config.X_test_file_path, config.y_test_file_path, B),
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
