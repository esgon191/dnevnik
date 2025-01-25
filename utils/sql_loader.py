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
                 y_columns: list):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.table = table
        self.batch_size = batch_size
        self.X_columns = X_columns
        self.y_columns = y_columns

        self.current = 0

        # Движок подключения к бд
        self.engine = self.connect()

        # Запрос уникальных батч айди для последующего итерирования по ним
        query = f"SELECT DISTINCT batch_id FROM {self.table};"
        self.batch_ids = pd.read_sql_query(query, self.engine)['batch_id'].values

        # Дополнительное перемешивание
        np.random.shuffle(self.batch_ids)

    def connect(self):        
        connection_string = f"postgresql+psycopg2://{dbconfig.POSTGRES_USER}:{dbconfig.POSTGRES_PASSWORD}@{dbconfig.HOST}:5432/{dbconfig.DB}"
        engine = create_engine(connection_string)
        return engine 
    
    def __iter__(self):
        """Итератор возвращает сам себя"""
        return self

    def __next__(self):
        """Следующий элемент итерации - данные для текущего batch_id"""
        X_batch = list()
        y_batch = list()
            
        for i in range(self.batch_size):
            if self.current < len(self.batch_ids):
                # Получаем текущий batch_id
                batch_id = self.batch_ids[self.current]
                self.current += 1

                # Выполняем запрос для текущего batch_id
                query = f"SELECT * FROM {self.table} WHERE batch_id = '{batch_id}';"
                result = pd.read_sql_query(query, self.engine)
            else:
                raise StopIteration

            X_chunk = result[self.X_columns].values
            y_chunk = result[self.y_columns].values[-1]

            X_chunk = X_chunk.reshape(1, 500, 10)        
            X_batch.append(X_chunk.copy())

            y_chunk = y_chunk.reshape(1, 1)
            y_batch.append(y_chunk.copy())


        X_batch = np.concatenate(X_batch, axis=0)
        y_batch = np.concatenate(y_batch, axis=0)

        X_batch = tuple(np.split(X_batch, axis=2, indices_or_sections=10))
        
        return (X_batch, y_batch)