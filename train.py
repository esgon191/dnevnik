import tensorflow as tf
from models.TCMH import TCMH
from utils.dataset_factory import sql_generator_dataset_factory 
import datetime

import config
import dbconfig
from utils.logging import logger_factory

# Logging setup
train_logger = logger_factory(
    name='train_logger',
    file='logs/train.log'
) 

# Настройка tensorflow на работу на 10 ядрах
# Ограничить количество потоков
tf.config.threading.set_inter_op_parallelism_threads(10)
tf.config.threading.set_intra_op_parallelism_threads(10)

# Создание модели
model = TCMH()

train_logger.info('Created model')

# Компиляция модели 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

train_logger.info('Compiled model')

# Сборка модели
model.summary()

train_logger.info('Model builded')

# Тренировочный датасет
train_logger.info('Traun dataset creation')
train_dataset, train_steps = sql_generator_dataset_factory(
    dbconfig,
    config, 
    'new_objects_assigned',
    stratification_attr_name='train_0_test_1_val_2',
    stratification_attr=0,
    batch_id_name='object_id'
)

# Валидационный датасет 
train_logger.info('Validation dataset creation')
val_dataset, val_steps = sql_generator_dataset_factory(
    dbconfig,
    config, 
    'new_objects_assigned',
    stratification_attr_name='train_0_test_1_val_2',
    stratification_attr=2,
    batch_id_name='object_id'
) 

# Обучение модели
train_logger.info('Learning started')
model.fit(
    train_dataset,
    epochs=config.EPOCHS,
    validation_data=val_dataset,  # передаем валидационный датасет
    steps_per_epoch=train_steps, # Шагов за эпоху
    validation_steps=val_steps # Валидационных шагов
)

train_logger.info('Learning ended')
name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#name = 'savedmodel'

model.save(f'./models/{name}.keras') 

# Тестирование модели 
train_logger.info('Testing started')

test_dataset, test_steps = sql_generator_dataset_factory(
    dbconfig, 
    config,
    'new_objects_assigned',
    stratification_attr_name='train_0_test_1_val_2',
    stratification_attr=1,
    batch_id_name='object_id'
)

loss, accuracy = model.evaluate(test_dataset)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')