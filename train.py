from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from pickle import dump

from config import DefaultConfig
from model import build_model
from data import prep_cifar100_datasets


def _train(config):
    BEST_FILE = f"models/{config.model_name}_best.keras"
    HISTORY_FILE = f"history/{config.model_name}.pkl"

    train, val, test = prep_cifar100_datasets(val_split=config.val_ratio,
                                              batch_size=config.batch_size)

    model = build_model(input_shape=(32, 32, 3), num_classes=100, blocks=config.blocks, dropouts=config.dropouts, kernel_size=config.kernel_size)

    model.compile(optimizer=Adam(learning_rate=config.lr),
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
        epochs=config.num_epochs,
        validation_data=val,
        verbose=1,
        callbacks=[checkpoint_callback],
    )

    best_model = load_model(BEST_FILE)

    test_loss, test_accuracy = best_model.evaluate(test, verbose=1)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    with open(HISTORY_FILE, 'wb') as f:
        dump(history.history, f)


if __name__ == "__main__":
    CONFIG = DefaultConfig().parse()
    _train(CONFIG)


def train(**kwargs):
    config = DefaultConfig().args(**kwargs)
    _train(config)
