import pandas as pd
import logging

# Logging setup
data_generator_logger = logging.getLogger(name='data_generator_logger')

data_generator_handler = logging.FileHandler('logs/train_generator.log', mode='w')
data_generator_handler.setLevel(logging.DEBUG)

data_generator_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
data_generator_handler.setFormatter(data_generator_formatter)

data_generator_logger.addHandler(data_generator_handler)

# Generator
def data_generator(X_file_path, y_file_path, sequence_length, x_columns, y_column):
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


        for X_chunk, y_chunk in zip(X_generator, y_generator):
            X_batch = tuple(X_chunk[column].values.reshape(-1, 500, 1) for column in X_chunk.columns)
            y_batch = y_chunk["coarse"].values
            yield (X_batch, y_batch)

            data_generator_logger.debug(f'Yeilded chunk {len(X_batch)}, {X_batch[0].shape}, {len(X_batch)}, {y_batch.shape}')