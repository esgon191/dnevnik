import pandas as pd
import numpy as np 
import logging
from utils.logging import logger_factory

# Logging setup
data_generator_logger = logger_factory(
    name='data_generator_logger',
    file='logs/train_genrator.log'
)


# Generator
def data_generator(X_file_path, y_file_path, x_columns, y_column, batch_size, sequence_length, overlap):
    #reads_per_chunk = sequence_length//overlap

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
            X_chunk = next(X_generator).values
            y_chunk = next(y_generator).values[-1]

            X_chunk = X_chunk.reshape(1, 500, 10)        
            X_batch.append(X_chunk.copy())

            y_chunk = y_chunk.reshape(1, 1)
            y_batch.append(y_chunk.copy())
            # yield X_chunk
            # X_chunk = readed
                
            
            
            # X_chunk = next(X_generator).values.reshape(1, 500, 10)
            #y_chunk = next(y_generator).values.reshape(1, 500, 1)

            #X_batch.append(X_chunk)
            #y_batch.append(y_chunk)

        X_batch = np.concatenate(X_batch, axis=0)
        y_batch = np.concatenate(y_batch, axis=0)

        X_batch = tuple(np.split(X_batch, axis=2, indices_or_sections=10))
        
        yield (X_batch, y_batch)

        data_generator_logger.debug('yileded chunk')