import config, dbconfig, argparse
from utils.dataset_factory import sql_generator_dataset_factory
import tensorflow as tf

# Использовать 10 ядер процессора
tf.config.threading.set_intra_op_parallelism_threads(10)
tf.config.threading.set_inter_op_parallelism_threads(10)

# Загрузка модели из параметров запуска файла
parser = argparse.ArgumentParser(description="Скрипт для тестирования модели")
parser.add_argument('--model', type=str, required=True, help="Путь к модели")
args = parser.parse_args()

print(f"Загружаем модель из {args.model}")
model = tf.keras.models.load_model(args.model)

test_dataset, test_steps = sql_generator_dataset_factory(
    dbconfig, 
    config,
    'test_std'
)

loss, accuracy = model.evaluate(test_dataset)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')