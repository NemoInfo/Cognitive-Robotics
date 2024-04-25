from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from pickle import dump

from config import DefaultConfig
from model import CustomCNN
from data import prep_cifar100_datasets

if __name__ == "__main__":
    CONFIG = DefaultConfig().parse()
    BEST_FILE = f"{CONFIG.model_name}_best.h5"

    train, val, test = prep_cifar100_datasets(val_split=CONFIG.val_ratio,
                                              batch_size=CONFIG.batch_size)

    model = CustomCNN(input_shape=(32, 32, 3), num_classes=100)

    model.compile(optimizer=Adam(learning_rate=CONFIG.lr),
                  loss=['categorical_crossentropy'],
                  metrics=['accuracy'])

    checkpoint_callback = ModelCheckpoint(
        BEST_FILE,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    history = model.fit(
        train,
        epochs=CONFIG.num_epochs,
        validation_data=val,
        verbose=1,
        callbacks=[checkpoint_callback],
    )

    best_model = load_model(BEST_FILE)

    test_loss, test_accuracy = best_model.evaluate(test, verbose=1)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    with open('history.pkl', 'wb') as f:
        dump(history.history, f)
