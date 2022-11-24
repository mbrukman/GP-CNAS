from __future__ import annotations
from src.pipeline import Pipeline
from src.tree import GPTree, Individual
from src.utils import check_model
import tensorflow as tf


class TFModelConvert(object):
    def __init__(
        self,
        pipeline: Pipeline,
        epoch: int,
        loss,
        optim: str,
        metrics: list[str],
    ):
        """
        Class that converts a GPTree object into a Keras model. Also, it train and evaluate each model.

        :param pipeline: Pipeline object to have access to the datasets
        :param epoch: Number of epochs
        :param loss: Keras loss funcion
        :param optim: Keras optimizer
        :param metrics: List of metrics to Tensorflow/Keras monitor
        """
        self.pipeline = pipeline
        self.epochs = epoch
        self.loss = loss
        self.optim = optim
        self.metrics = metrics

    def build_model(self, layers: list):
        """
        Method that using a list of Keras layers create a Sequential Model, compile it base on loss, optimizer and
        metrics and in the end show the summary.

        :param layers: List of Keras layers
        :return: Compiled Keras model
        """
        layers += [
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(self.pipeline.out_size),
        ]

        model = tf.keras.Sequential(layers)

        model.compile(
            optimizer=self.optim,
            loss=self.loss(from_logits=True),
            metrics=self.metrics,
        )

        model.summary()

        return model

    def create_model(self, tree: GPTree):
        """
        Method responsible for evalueate a tree into a list of Resblocks and then create a list of layers
        starting with a Input layer with the right data shape.

        :param tree: GPTree object of an individual
        :return: List of Keras layers
        """
        models = tree.eval_tree()

        model_layers = [tf.keras.Input(shape=self.pipeline.in_size)]
        model_layers += models

        return model_layers

    def fit_model(self, individual: Individual):
        """
        Method wich trains a model after creating and building it.

        **Warning**: The check_model() function is used to make sure that the tree generates
        a valid model (with a valid output shape). If the model isn't valid it will truncate to become valid.
        Although, the GPTree object is not changed because map a GPTree to a Keras model is easy but the inverse
        isn't that easy.

        :param individual:
        :return:
        """
        # Create Keras layers
        layers = self.create_model(individual.tree)

        # Validate if layers has valid output shape, if not trim it
        index, _ = check_model(layers)
        layers = layers[:index]

        # Build model
        tf_model = self.build_model(layers)

        # Train model
        tf_model.fit(
            self.pipeline.train_ds,
            epochs=self.epochs,
            validation_data=self.pipeline.valid_ds,
        )

        # Evaluate model
        loss, acc = tf_model.evaluate(self.pipeline.test_ds, verbose=0)

        print("Model eval - Loss: %.3f | Acc: %.3f\n" % (loss, acc))

        return loss, acc
