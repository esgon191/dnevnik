import config, dbconfig
from utils.sql_loader import SqlLoader
import tensorflow as tf
from models import TCMH

sql_iter_instance = SqlLoader(
    host=dbconfig.HOST,
    port='5432',
    database=dbconfig.DB,
    user=dbconfig.POSTGRES_USER,
    password=dbconfig.POSTGRES_PASSWORD,
    table='test_std',
    batch_size=config.BATCH_SIZE,
    X_columns=config.X_COLUMNS,
    y_columns=config.Y_COLUMN
)

output_signature = (
    tuple(tf.TensorSpec(shape=(config.BATCH_SIZE, *config.INPUT_SHAPE), dtype=tf.float32) for _ in range(10)),
    tf.TensorSpec(shape=(config.BATCH_SIZE, 1), dtype=tf.int32),
)

tf.config.threading.set_intra_op_parallelism_threads(10)
tf.config.threading.set_inter_op_parallelism_threads(10)

saved_weights = input("Название модели: ") # Путь к модели относительно этого файла

# Восстановление весов модели
model = TCMH()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.load_weights(saved_weights)

data_handler = lambda : iter(sql_iter_instance)

test_dataset = tf.data.Dataset.from_generator(
    data_handler,
    output_signature=output_signature
).prefetch(tf.data.experimental.AUTOTUNE)

loss, accuracy = model.evaluate(test_dataset)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')