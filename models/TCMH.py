import tensorflow as tf
from tensorflow.keras import layers, models
from utils.logging import logger_factory

class TCMH(models.Model):
    def __init__(self, input_shape=None, num_sensors=10, num_heads=8, filters=32, output_units=9, **kwargs):
        # Передаем kwargs в базовый класс
        super(TCMH, self).__init__()
        
        # Если input_shape передан как TensorSpec, преобразуем его в кортеж
        if isinstance(input_shape, tf.TensorSpec):
            input_shape = tuple(input_shape.shape.as_list())
        
        # Сохранение параметров модели (теперь input_shape — кортеж, пригодный для сериализации)
        self.input_shape_param = input_shape
        self.num_sensors = num_sensors
        self.num_heads = num_heads
        self.filters = filters
        self.output_units = output_units

        # Если нужны входные слои (хотя они здесь не используются для вычислений),
        # можно создать их для удобства (они не включаются в конфигурацию модели)
        self.input_layers = [layers.Input(shape=input_shape) for _ in range(num_sensors)]
        
        # Temporal Convolutional Layers для каждого датчика
        self.conv1_layers = [
            layers.Conv1D(filters=filters, kernel_size=3, padding='causal', dilation_rate=1, activation='relu')
            for _ in range(num_sensors)
        ]
        self.conv2_layers = [
            layers.Conv1D(filters=filters, kernel_size=3, padding='causal', dilation_rate=2, activation='relu')
            for _ in range(num_sensors)
        ]
        self.pool_layers = [layers.MaxPooling1D(pool_size=2) for _ in range(num_sensors)]
        
        # Multi-Head Attention Layer
        self.multi_head_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=filters)
        
        # Дополнительные слои
        self.conv_layer = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')
        self.max_pool = layers.MaxPooling1D(pool_size=2)
        self.global_avg_pool = layers.GlobalAveragePooling1D()
        self.output_layer = layers.Dense(units=output_units, activation='softmax')
        
    def call(self, inputs, training=None, **kwargs):
        conv_outputs = []
        for i in range(self.num_sensors):
            x = self.conv1_layers[i](inputs[i])
            x = self.conv2_layers[i](x)
            x = self.pool_layers[i](x)
            conv_outputs.append(x)
        
        concat_layer = layers.Concatenate(axis=-1)(conv_outputs)
        
        x = self.multi_head_attention(concat_layer, concat_layer)
        
        x = self.conv_layer(x)

        x = self.max_pool(x)
        
        x = self.global_avg_pool(x)
        
        return self.output_layer(x)
    
    def get_config(self):
        # Получаем базовую конфигурацию модели
        config = super(TCMH, self).get_config()
        # Обновляем конфигурацию только параметрами, необходимыми для восстановления модели.
        config.update({
            "input_shape": self.input_shape_param,  # здесь кортеж, а не TensorSpec
            "num_sensors": self.num_sensors,
            "num_heads": self.num_heads,
            "filters": self.filters,
            "output_units": self.output_units,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        # Восстанавливаем модель по конфигурации
        return cls(**config)

    @classmethod
    def build_from_config(cls, config):
        # Создаем модель из конфигурации
        model = cls(**config)
        # Вызываем build с правильной входной формой:
        # Предполагается, что config['input_shape'] — это кортеж, описывающий форму одного примера.
        model.build((None, *config["input_shape"]))
        return model