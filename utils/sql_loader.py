import numpy as np
import pandas as pd
import config
import dbconfig
from sqlalchemy import create_engine

np.random.seed(42)

class SqlLoader:
    def __init__(self, 
                 host : str, 
                 port : (int | str), 
                 database : str, 
                 user : str, 
                 password : str,
                 table : str,
                 batch_size : int,
                 X_columns : list,
                 y_columns: list,
                 batch_id_name='batch_id',
                 stratification_attr_name=None,
                 stratification_attr=None):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.table = table
        self.batch_size = batch_size
        self.X_columns = X_columns
        self.y_columns = y_columns
        # По какому полю столбцу определен конкретный обучающий объект
        self.batch_id_name = batch_id_name
        # Дополнительный фильтр, по которому производится заброс к нужной выборке из таблицы
        self.stratification_attr_name = stratification_attr_name
        # Значение фильтра выборки
        self.stratification_attr = stratification_attr 

        self.current = 0

        # Движок подключения к бд
        self.engine = self.connect()

        # Формируем часть запроса с фильтром на выборку
        strat_part = ''
        if self.stratification_attr_name is not None:
            strat_part = f'WHERE {self.stratification_attr_name} = {self.stratification_attr}'

        # Запрос уникальных батч айди для последующего итерирования по ним
        query = f"SELECT DISTINCT {self.batch_id_name} FROM {self.table} {strat_part};"
        self.batch_ids = pd.read_sql_query(query, self.engine)[self.batch_id_name].values

        # Количество обучающих примеров в датасете
        self.amount_of_samples = len(self.batch_ids)

        # Шагов за проход генератора
        self.steps_per_epoch = self.__steps_per_epoch()

    def connect(self):        
        connection_string = f"postgresql+psycopg2://{dbconfig.POSTGRES_USER}:{dbconfig.POSTGRES_PASSWORD}@{dbconfig.HOST}:5432/{dbconfig.DB}"
        engine = create_engine(connection_string)
        return engine 
    
    def __iter__(self):
        """Итератор возвращает сам себя, предварительно перемешивая данные и обновляя счетчик"""
        self.current = 0
        np.random.shuffle(self.batch_ids)
        return self

    def __next__(self):
        """Следующий элемент итерации - данные для текущего batch_id"""
        X_batch = list()
        y_batch = list()
            
        if self.current < self.steps_per_epoch:
            for step_in_batch in range(self.batch_size):
                # Получаем текущий batch_id
                # Так как current - счетчик шагов за эпоху, то номер объекта вычисляется как:
                # Текущий шаг за эпоху * размер батча + текущий объект внутри формируемого батча
                batch_id = self.batch_ids[self.current * self.batch_size + step_in_batch]

                # Выполняем запрос для текущего batch_id
                query = f"SELECT * FROM {self.table} WHERE {self.batch_id_name} = {batch_id};"
                result = pd.read_sql_query(query, self.engine)

                X_chunk = result[self.X_columns].values
                y_chunk = result[self.y_columns].values[-1]

                X_chunk = X_chunk.reshape(1, 500, 10)        
                X_batch.append(X_chunk.copy())

                y_chunk = y_chunk.reshape(1, 1)
                y_batch.append(y_chunk.copy())

            self.current += 1
                
        else:
            raise StopIteration


        X_batch = np.concatenate(X_batch, axis=0)
        y_batch = np.concatenate(y_batch, axis=0)

        X_batch = tuple(np.split(X_batch, axis=2, indices_or_sections=10))
        
        return (X_batch, y_batch)

    def __steps_per_epoch(self):
        """
        Возвращает количество отдаваемых батчей один проход генератора 
        Один проход генератора = одна эпоха
        """
        return self.amount_of_samples // self.batch_size 