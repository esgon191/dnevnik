import tensorflow as tf
from tensorflow.keras import layers, models
from utils.logging import logger_factory

class TCMH(models.Model):
    def __init__(self, input_shape=None, logger=None, num_sensors=10, num_heads=8, filters=32, output_units=9, **kwargs):
        super(TCMH, self).__init__()

        # Сохранение параметров модели
        self.input_shape_param = input_shape
        self.num_sensors = num_sensors
        self.num_heads = num_heads
        self.filters = filters
        self.output_units = output_units

        if logger == None:
            self.logger = logger_factory(
                name='model_logger',
                file='logs/model.log'
            )
        else:
            self.logger = logger

        print(input_shape)

        self.num_sensors = num_sensors
        self.input_layers = [layers.Input(shape=input_shape) for _ in range(num_sensors)]
        
        # Temporal Convolutional Layers
        self.conv1_layers = [layers.Conv1D(filters=filters, kernel_size=3, padding='causal', dilation_rate=1, activation='relu') for _ in range(num_sensors)]
        self.conv2_layers = [layers.Conv1D(filters=filters, kernel_size=3, padding='causal', dilation_rate=2, activation='relu') for _ in range(num_sensors)]
        self.pool_layers = [layers.MaxPooling1D(pool_size=2) for _ in range(num_sensors)]
        
        # Multi-Head Attention Layer
        self.multi_head_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=filters)
        
        # Convolutional Layer
        self.conv_layer = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')
        self.max_pool = layers.MaxPooling1D(pool_size=2)
        
        # Global Average Pooling
        self.global_avg_pool = layers.GlobalAveragePooling1D()
        
        # Output Layer
        self.output_layer = layers.Dense(units=output_units, activation='softmax')
        self.logger.info('initialized')

    def __call__(self, inputs):
        conv_outputs = []
        self.logger.debug('conv1d')
        for i in range(self.num_sensors):
            x = self.conv1_layers[i](inputs[i])
            x = self.conv2_layers[i](x)
            x = self.pool_layers[i](x)
            conv_outputs.append(x)
        
        self.logger.debug('concatenation')
        concat_layer = layers.Concatenate(axis=-1)(conv_outputs)
        
        # Multi-Head Attention
        self.logger.debug('multi-head attention')
        x = self.multi_head_attention(concat_layer, concat_layer)
        
        # Convolutional Layer
        self.logger.debug('convolution')
        x = self.conv_layer(x)

        self.logger.debug('max pooling')
        x = self.max_pool(x)
        
        # Global Average Pooling
        self.logger.debug('global average pooling')
        x = self.global_avg_pool(x)
        
        # Output Layer
        self.logger.info('returning')
        return self.output_layer(x)
    
    def get_config(self):
        config = super(TCMH, self).get_config()
        config.update({
            "input_shape": self.input_shape_param,
            "num_sensors": self.num_sensors,
            "num_heads": self.num_heads,
            "filters": self.filters,
            "output_units": self.output_units,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

