from tensorflow.keras.datasets import cifar100
from tensorflow.keras import utils
from tensorflow.data import Dataset, AUTOTUNE
from typing import Tuple


def prep_cifar100_datasets(val_split: float = 0.2, batch_size: int = 32) -> Tuple[Dataset, Dataset, Dataset]:
    (train_samples, train_labels), (test_samples, test_labels) = cifar100.load_data()

    train_labels = utils.to_categorical(train_labels, 100)
    test_labels = utils.to_categorical(test_labels, 100)

    train_samples = train_samples.astype('float32') / 255
    test_samples = test_samples.astype('float32') / 255

    train_dataset = Dataset.from_tensor_slices((train_samples, train_labels))
    train_dataset = train_dataset.shuffle(buffer_size=1000)

    test_dataset = Dataset.from_tensor_slices((test_samples, test_labels))

    num_val_samples = int(val_split * len(train_samples))
    val_dataset = train_dataset.take(num_val_samples)
    train_dataset = train_dataset.skip(num_val_samples)

    train_dataset = train_dataset.batch(batch_size).prefetch(AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(AUTOTUNE)

    return train_dataset, val_dataset, test_dataset
