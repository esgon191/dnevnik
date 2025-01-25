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
                 table : str):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.table = table
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
        if self.current < len(self.batch_ids):
            # Получаем текущий batch_id
            batch_id = self.batch_ids[self.current]
            self.current += 1

            # Выполняем запрос для текущего batch_id
            query = f"SELECT * FROM {self.table} WHERE batch_id = '{batch_id}';"
            result = pd.read_sql_query(query, self.engine)

            return result  # Возвращаем результат запроса в виде DataFrame
        else:
            raise StopIteration