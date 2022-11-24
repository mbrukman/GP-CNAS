import tensorflow as tf
import tensorflow_datasets as tfds
import os


class Pipeline(object):
    def __init__(self, dataset_name, batch_size=32):
        """
        Class that download, process and create 3 datatsets:
            - Train dataset (80%)
            - Validation dataset (10%)
            - Test dataset (10%)

        Any dataset included in the module tensorflow-datasets can be used.

        :param dataset_name: Name of dataset inside the tensorflow-datasets
        :param batch_size: Number of batches to be processed in each train/inference iteration
        """
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.in_size = self.out_size = None

        # Download Dataset
        train, val, test = self.download_dataset()

        # Preprocess Datasets
        self.train_ds, self.valid_ds, self.test_ds = map(
            lambda x: self.preprocess_dataset(x, batch_size),
            [train, val, test],
        )

    def download_dataset(self):
        """
        Method that download the dataset and split it in 3 parts. Also, it saves the input size and the number of labels
        because they're needed in other parts of the code.
        :return: Tuple of three tf.data.Dataset objects
        """
        (train_ds, valid_ds, test_ds), ds_info = tfds.load(
            self.dataset_name,
            split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
            as_supervised=True,
            with_info=True,
            data_dir=os.path.abspath(os.path.dirname("../../tfds_datasets/")),
        )

        self.in_size = ds_info.features["image"].shape
        self.out_size = ds_info.features["label"].num_classes

        return train_ds, valid_ds, test_ds

    def preprocess_dataset(
        self, ds: tf.data.Dataset, batch_size: int
    ) -> tf.data.Dataset:
        """
        Method that preprocess a dataset to make sure it is normalized, batched, shuffled and cached

        :param ds: tf.data.Dataset object representing a dataset
        :param batch_size: Bacth size
        :return: Processed tf.data.Dataset object
        """
        # Normalize Dataset [0, 255] -> [0,1]
        normalizer = tf.keras.layers.Rescaling(1.0 / 255)

        return (
            ds.batch(batch_size)
            .shuffle(tf.data.experimental.cardinality(ds))
            .map(lambda x, y: (normalizer(x), y))
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )
