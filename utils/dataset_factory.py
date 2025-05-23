from utils.sql_loader import SqlLoader
import tensorflow as tf

def sql_generator_dataset_factory(
            dbconfig, # Конфиг подключения к базе данных
            config, # Тренировочный конфиг
            table : str, # Таблица, откуда брать данные 
            batch_id_name='batch_id',
            stratification_attr_name=None,
            stratification_attr=None
            ) -> tuple[tf.data.Dataset, int]: # Датасет и количество шагов за датасет
    """
    Фабрика для создания датасетов из SqlLoader на основе tf.data.Dataset.from_generator
    """

    # Инстанс SqlLoader
    sql_iter_instance = SqlLoader(
        host=dbconfig.HOST,
        port=dbconfig.PORT,
        database=dbconfig.DB,
        user=dbconfig.POSTGRES_USER,
        password=dbconfig.POSTGRES_PASSWORD,
        table=table,
        batch_size=config.BATCH_SIZE,
        X_columns=config.X_COLUMNS,
        y_columns=config.Y_COLUMN,
        batch_id_name=batch_id_name,
        stratification_attr_name=stratification_attr_name,
        stratification_attr=stratification_attr
    )

    # Сколько шагов за один проход по датасету получит модель 
    # обучающих объектов // размер батча
    steps_per_epoch = sql_iter_instance.steps_per_epoch

    # На каждой новой эпохе tf пытается получить новый генератор через метод __call__()
    # В будущем перенесу это прямо в функционал SqlLoader
    data_handler = lambda : iter(sql_iter_instance)

    # Выходная форма генератора 
    output_signature = (
        tuple(tf.TensorSpec(shape=(config.BATCH_SIZE, *config.INPUT_SHAPE), dtype=tf.float32) for _ in range(10)),
        tf.TensorSpec(shape=(config.BATCH_SIZE, 1), dtype=tf.int32),
    )

    # Создание датасета 
    dataset = tf.data.Dataset.from_generator(
        data_handler,
        output_signature=output_signature,
    ).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset, steps_per_epoch