from models.model import Model
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam

class TransferedModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your existing code...
        basic_model = models.load_model('results/basic_model_10_epochs_timestamp_1708659836.keras')
        for layer in basic_model.layers:
            layer.trainable = False

        self.model = Sequential()
        for i, layer in enumerate(basic_model.layers[:-1]):
            # Provide unique names for Dense and Dropout layers
            if isinstance(layer, layers.Dense):
                self.model.add(layers.Dense(64, activation='relu', name=f'dense_{i}'))
            elif isinstance(layer, layers.Dropout):
                self.model.add(layers.Dropout(0.5, name=f'dropout_{i}'))
            else:
                self.model.add(layer)

        self.model.add(layers.Dense(categories_count, activation='softmax'))
    
    def _compile_model(self):
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
