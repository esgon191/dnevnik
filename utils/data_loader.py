import pandas as pd
import numpy as np 
import logging

# Logging setup
data_generator_logger = logging.getLogger(name='data_generator_logger')

data_generator_handler = logging.FileHandler('logs/train_generator.log', mode='w')
data_generator_handler.setLevel(logging.DEBUG)

data_generator_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
data_generator_handler.setFormatter(data_generator_formatter)

data_generator_logger.addHandler(data_generator_handler)

# Generator
def data_generator(X_file_path, y_file_path, batch_size, sequence_length, x_columns, y_column):
    while True:
        X_generator = pd.read_csv(
            X_file_path, 
            chunksize=sequence_length,
            usecols=x_columns
            )
        y_generator = pd.read_csv(
            y_file_path, 
            chunksize=sequence_length,
            usecols=y_column
            )

        X_batch = list()
        y_batch = list()

        for i in range(batch_size):
            X_chunk = next(X_generator).values.reshape(1, 500, 10)
            y_chunk = next(y_generator).values.reshape(1, 500, 1)

            X_batch.append(X_chunk)
            y_batch.append(y_chunk)

        X_batch = np.concatenate(X_batch, axis=0)
        y_batch = np.concatenate(y_batch, axis=0)

        X_batch = tuple(np.split(X_batch, axis=2, indices_or_sections=10))
        
        yield (X_batch, y_batch)

        data_generator_logger.debug('yileded chunk')