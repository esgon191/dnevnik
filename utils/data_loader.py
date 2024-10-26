def data_generator(file_paths, batch_size):
    while True:
        for file_path in file_paths:
            for chunk in pd.read_csv(file_path, chunksize=batch_size):
                # Assuming the data has columns 'sensor_0', 'sensor_1', ..., 'sensor_9' and 'labels'
                X_batch = [chunk[f"sensor_{i}"].values.reshape(-1, 500, 1) for i in range(10)]
                y_batch = chunk["labels"].values
                yield X_batch, y_batch