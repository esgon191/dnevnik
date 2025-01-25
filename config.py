BATCH_SIZE = 32
SEQUENCE_LENGTH = 500
OVERLAP = 0 # SEQUENCE_LENGTH % OVERLAP = 0, OVERLAP < SEQUENCE_LENGTH 
EPOCHS = 10
INPUT_SHAPE = (SEQUENCE_LENGTH, 1)
X_COLUMNS = [
    'acceleration_x', 'acceleration_y', 'acceleration_z', 
    'gyro_x', 'gyro_y', 'gyro_z', 
    'magnetometer_x', 'magnetometer_y', 'magnetometer_z', 
    'pressure'
]

Y_COLUMN = ['coarse']

X_train_file_path = "data/pipline_test/X_train.csv"
y_train_file_path = "data/pipline_test/y_train.csv"

X_val_file_path = "data/pipline_test/X_val.csv"
y_val_file_path = "data/pipline_test/y_val.csv"

X_test_file_path = "data/pipline_test/X_test.csv"
y_test_file_path = "data/pipline_test/y_test.csv"