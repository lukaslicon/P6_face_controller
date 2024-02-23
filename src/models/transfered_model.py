from models.model import Model
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam

class TransferedModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Load your trained facial recognition model
        basic_model = models.load_model('results/basic_model_100_epochs_timestamp_1708669386.keras')

        # Eliminate the final softmax layer (which specializes it to a 3-class problem)
        basic_model.pop()

        # Freeze all parameters in the remainder so that they cannot change through further learning
        for layer in basic_model.layers:
            layer.trainable = False

        # Bolt on one or more fully connected layers to perform an arbitrary computation
        self.model = Sequential([
            basic_model,
            layers.Dense(64, activation='relu', name='dense_layer'),
        ])

        # Add a softmax layer for the new task
        self.model.add(layers.Dense(categories_count, activation='softmax', name='output_layer'))

    def _compile_model(self):
        # Compile the model
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

    def _train_model(self, train_data, val_data, epochs):
        # Train only the new FCNN and softmax layers on data from the new task
        history = self.model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
        )
        return history
