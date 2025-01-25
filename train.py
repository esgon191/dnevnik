import logging.config
import tensorflow as tf
from models.TCMH import TCMH
from utils.data_loader import data_generator
import utils.data_loader
from utils.sql_loader import SqlLoader

import importlib
importlib.reload(utils.data_loader)

import config
import dbconfig
from utils.logging import loggerFactory

# Logging setup
train_logger = loggerFactory(
    name='train_logger',
    file='logs/train.log'
) 

model_logger = loggerFactory(
    name='model_logger',
    file='logs/model.log'
)

# Define the model
B = config.BATCH_SIZE  # Batch size 
model = TCMH(
    input_shape=config.INPUT_SHAPE,
    logger=model_logger
)

train_logger.info('Created model')

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

train_logger.info('Compiled model')

# Summary of the model
model.build([tf.TensorShape((500, 1)) for _ in range(10)])
model.summary()

train_logger.info('Model builded')

output_signature = (
    tuple(tf.TensorSpec(shape=(config.BATCH_SIZE, *config.INPUT_SHAPE), dtype=tf.float32) for _ in range(10)),
    tf.TensorSpec(shape=(config.BATCH_SIZE, 1), dtype=tf.int32),
)

sql_iter_instance = SqlLoader(
    host=dbconfig.HOST,
    port='5432',
    database=dbconfig.DB,
    user=dbconfig.POSTGRES_USER,
    password=dbconfig.POSTGRES_PASSWORD,
    table='train_std',
    batch_size=config.BATCH_SIZE,
    X_columns=config.X_COLUMNS,
    y_columns=config.Y_COLUMN
)

data_handler = lambda : iter(sql_iter_instance)

train_logger.info('Dataset creation')
# Create Dataset
train_dataset = tf.data.Dataset.from_generator(
    data_handler,
    output_signature=output_signature
).prefetch(tf.data.experimental.AUTOTUNE)

# Training the model
train_logger.info('Learning started')
model.fit(train_dataset, epochs=config.EPOCHS, steps_per_epoch=len(config.X_train_file_path) // B)

train_logger.info('Learning ended')
model.save('.')
# Evaluation of the model
train_logger.info('Testing started')
test_dataset = tf.data.Dataset.from_generator(
    data_handler,
    output_signature
).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

loss, accuracy = model.evaluate(test_dataset, steps=len(config.test_file_paths) // B)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Make predictions
predictions = model.predict(test_dataset, steps=len(config.test_file_paths) // B)
print('Predictions:', predictions)
