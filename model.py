from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, RandomZoom, RandomTranslation

class CustomCNN(Sequential):
    def __init__(self, input_shape, num_classes):
        super().__init__()

        self.add(RandomFlip("horizontal_and_vertical", input_shape=input_shape))
        self.add(RandomRotation(0.1))
        self.add(RandomZoom(0.1))
        self.add(RandomTranslation(0.1, 0.1))

        # 1st block
        self.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        self.add(Activation('relu'))
        self.add(BatchNormalization())
        self.add(Conv2D(32, (3, 3), padding='same'))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.2))

        # 2nd block
        self.add(Conv2D(64, (3, 3), padding='same'))
        self.add(Activation('relu'))
        self.add(BatchNormalization())
        self.add(Conv2D(64, (3, 3), padding='same'))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.3))

        # 3rd block
        self.add(Conv2D(128, (3, 3), padding='same'))
        self.add(Activation('relu'))
        self.add(BatchNormalization())
        self.add(Conv2D(128, (3, 3), padding='same'))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.4))

        # Dense
        self.add(Flatten())
        self.add(Dense(512))
        self.add(Activation('relu'))
        self.add(Dropout(0.5))
        self.add(BatchNormalization())

        # Output
        self.add(Dense(num_classes))
        self.add(Activation('softmax'))
