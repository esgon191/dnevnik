import config, dbconfig
from utils.sql_loader import SqlLoader
import tensorflow as tf

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

savedmodel = input("Название модели: ") # Путь к модели относительно этого файла

model = tf.keras.models.load_model(savedmodel)

data_handler = lambda : iter(sql_iter_instance)

test_dataset = tf.data.Dataset.from_generator(
    data_handler,
    output_signature=output_signature
).prefetch(tf.data.experimental.AUTOTUNE)

test_dataset = test_dataset.take(10000)

@tf.function
def evaluate_model(model, dataset, steps=1000):
    return model.evaluate(dataset, steps=steps)

loss, accuracy = evaluate_model(model, test_dataset)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')