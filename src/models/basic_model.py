from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.optimizers import RMSprop, Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        self.model = Sequential([
            Rescaling(1./255, input_shape=input_shape),
            #layer 1
            layers.Conv2D(4, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            #layer 2
            layers.Conv2D(8, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            #layer 3
            layers.Conv2D(16, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            Dropout(0.3),

            #layer 4
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            Dropout(0.4),

            #flatten layers
            layers.Flatten(),

            #connected layers
            layers.Dense(64, activation='relu'),
            layers.Dense(categories_count, activation='softmax')
        ])
    
    def _compile_model(self):
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )