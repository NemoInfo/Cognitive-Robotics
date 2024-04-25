from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, RandomZoom, RandomTranslation


def build_model(input_shape, num_classes, blocks):
    model = Sequential()
    model.add(RandomFlip("horizontal_and_vertical", input_shape=input_shape))
    model.add(RandomRotation(0.1))
    model.add(RandomZoom(0.1))
    model.add(RandomTranslation(0.1, 0.1))

    # Convolution Blocks
    for block in blocks:
        model.add(Conv2D(block, (3, 3), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(block, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

    # Dense
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    # Output
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model
