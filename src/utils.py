import tensorflow as tf
from functools import partial
from typing import Union


def check_model(layers: list) -> tuple[int, tuple]:
    """
    Function that verifies if a model as small enough output size.
    If it reached zero or negative sizes should be truncated.

    The index where it should be truncated is returned as well as
    the output size of the model.

    :param layers: List of Keras layers
    :return: Tuple with the index to truncate and the output shape
    """
    in_shape = layers[0].shape

    for i in range(1, len(layers)):
        in_shape = layers[i].compute_output_shape(in_shape)

        # Can't have any more layers
        if in_shape[1] <= 1 or in_shape[2] <= 1:
            return i, in_shape

    return len(layers), in_shape


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, layers: list, *args, **kwargs):
        """
        Class that implements a Resnet block like those showed on the article.

        Normally the shortcut has a linear transformation layer to down sample
        the input to match the output.

        Although in the article nothing is showed, so I assumed the shortcut is directly connect to the output.
        To down sample the input was used a Max-Pooling calculated based on input and output size.

        To match the filters size was done the difference and zero padded the input or the output. In the
        end input and output are added.

        :param layers: List of Keras layers
        :param args: List of extra arguments
        :param kwargs: Dictionary of extra arguments
        """
        super(ResBlock, self).__init__(*args, **kwargs)
        self.layers = layers

    def get_config(self):
        """
        Override of get_config method from Keras. This method is used to tell to Keras what new attributes
        were implemented which can be important to serialize the layer.

        :return: Config variable with the updated attributes
        """
        config = super().get_config()
        config.update({"layers": self.layers})
        return config

    @tf.function
    def shortcut(self, inputs, x):
        """
        This method implements the shortcut like was explained above.

        Original input is recevied

        :param inputs: Original input tensor
        :param x: Output tensor
        :return: Add Keras layer adding input with output
        """
        in_shape = inputs.get_shape().as_list()
        x_shape = x.get_shape().as_list()

        kernel = (in_shape[1] - x_shape[1] + 1, in_shape[2] - x_shape[2] + 1)

        pool = tf.nn.max_pool2d(inputs, kernel, strides=1, padding="VALID")

        channels = abs(in_shape[-1] - x_shape[-1])

        # If pool as more channels than output, pad input
        # pool(same_height, same_width, more_channels), x(same_height, same_width, less_channels)
        if in_shape[-1] > x_shape[-1]:
            zeros = tf.zeros(shape=tf.shape(pool))
            zeros = zeros[:, :, :, :channels]
            x = tf.concat([x, zeros], axis=-1)

        # If pool as less channels than output, pad it
        # pool(same_height, same_width, less_channels), x(same_height, same_width, more_channels)
        elif in_shape[-1] < x_shape[-1]:
            zeros = tf.zeros(shape=tf.shape(x))
            zeros = zeros[:, :, :, :channels]
            pool = tf.concat([pool, zeros], axis=-1)

        return tf.keras.layers.Add(trainable=False)([pool, x])

    def compute_output_shape(self, input_shape: tuple):
        """
        Override Keras method to compute the shape of this custom layer.

        Since check_model() function exists, it was used to calculate the output shape of the ResBlock.

        The first layer is always a Input layer and the tuple doesn't contain the batch size to keep out
        of troubles with Tensorflow and dynamic sizing of tensors.

        :param input_shape: Tuple with the size of the input tensor
        :return: Tuple with the output shape
        """
        _, out_shape = check_model(
            [tf.keras.Input(shape=input_shape[1:])] + self.layers
        )

        return out_shape

    def call(self, inputs):
        """
        Override Keras method that is call in every passing through this layer.
        :param inputs:
        :return:
        """
        x = self.layers[0](inputs)

        for layer in self.layers[1:]:
            x = layer(x)

        # Add transformed input with output, x
        return self.shortcut(inputs, x)


class StridedConv2D(tf.keras.layers.Conv2D):
    def __init__(self, *args, **kwargs):
        """
        Class that is a custom type of 2D Convolutions, then could be distiguished to apply operations
        :param args: List of extra parameters
        :param kwargs: Dicitionary of extra parameters
        """
        super(StridedConv2D, self).__init__(*args, **kwargs)


def resnet_block1():
    """
    Function that constructs the b1 block showed on article
    :return: Resblock object initialized
    """
    return ResBlock(
        [
            tf.keras.layers.BatchNormalization(),
            StridedConv2D(
                16,
                (3, 3),
                strides=1,
                activation="relu",
                padding="valid",
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(
                16, (3, 3), strides=1, activation="relu", padding="valid"
            ),
        ]
    )


def resnet_block2():
    """
        Function that constructs the b2 block showed on article
        :return: Resblock object initialized
        """
    return ResBlock(
        [
            tf.keras.layers.BatchNormalization(),
            StridedConv2D(
                16,
                (3, 3),
                strides=1,
                activation="relu",
                padding="valid",
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(
                16, (3, 3), strides=1, activation="relu", padding="valid"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(
                16, (3, 3), strides=1, activation="relu", padding="valid"
            ),
        ]
    )


def resnet_block3():
    """
        Function that constructs the b3 block showed on article
        :return: Resblock object initialized
        """
    return ResBlock(
        [
            tf.keras.layers.BatchNormalization(),
            StridedConv2D(
                16,
                (3, 3),
                strides=1,
                activation="relu",
                padding="valid",
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(
                16, (1, 1), strides=1, activation="relu", padding="valid"
            ),
        ]
    )


def resnet_block4():
    """
        Function that constructs the b4 block showed on article
        :return: Resblock object initialized
        """
    return ResBlock(
        [
            tf.keras.layers.BatchNormalization(),
            StridedConv2D(
                16,
                (3, 3),
                strides=1,
                activation="relu",
                padding="valid",
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(
                16, (1, 1), strides=1, activation="relu", padding="valid"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(
                16, (3, 3), strides=1, activation="relu", padding="valid"
            ),
        ]
    )


def concat(t1: Union[list, ResBlock], t2: Union[list, ResBlock]):
    """
    Function that makes the concatenation (non-terminal node) of 2 model blocks (resnet blocks)

    :param t1: Object Resblock (terminal)
    :param t2: Another object Resblock (terminal)
    :return: List with Resblock objects inside
    """
    t_lists = list()

    for model in [t1, t2]:
        if isinstance(model, list):
            for k in model:
                t_lists.append(k)
        else:
            t_lists.append(model)

    return t_lists


def stride_factor(t):
    """
    Function that doubles the stride (non-terminal node) of a StrideConv2D
    :param t: ResBlock object (terminal)
    :return: List of Resblock objects
    """
    return apply_ops(t, partial(apply_stride, 2), "strides", (StridedConv2D,))


def two_times_filters(node):
    """
    Function that doubles the filter size (non-terminal node) of Convolutions
    :param node: Any non-terminal node
    :return: List of Resblock objects
    """
    return x_times_filters(node, 2)


def three_times_filters(node):
    """
    Function that triples the filter size (non-terminal node) of Convolutions
    :param node: Any non-terminal node
    :return: List of Resblock objects
    """
    return x_times_filters(node, 3)


def x_times_filters(node, x: int):
    """
    Function that X times the filter size (non-terminal node) of Convolutions

    If the node is another x_times_filters non-terminal node, then they sum up (as explained on the article)

    If the node is a list means some Resblocks are concatenated, then apply this primitive to the left most
    Resblock.

    :param x: Number of X times
    :param node: Any non-terminal node
    :return: List of Resblock objects
    """
    types = (tf.keras.layers.Conv2D, StridedConv2D)

    if isinstance(node, int):
        return node + x

    elif isinstance(node, list):
        t = apply_ops(node[0], partial(apply_filters, x), "filters", types)
        return [t] + node[1:]

    return apply_ops(node, partial(apply_filters, x), "filters", types)


def apply_ops(model, op_func: partial, attr: str, types: tuple):
    """
    Function that applies an operation:
     - apply_filters in case of x_times_filtesr
     - apply_strides in case of stride_factor

    To apply those functions were used partial functions to specify, without running, the factor of strides or
    filtes to be multiplied.

    In this fucntion those partial functions are executed, passing the current value
    (since the factor was passed previoulys on other functions) and applying the operations.

    :param model: Node to apply the factor
    :param op_func: Function that calculates the new attribute value based on the factor
    :param attr: Attribute which value will change
    :param types: Valid layer types to be applied the operation
    :return: Modified node
    """
    for layer in model.layers:
        if isinstance(layer, types):
            val = getattr(layer, attr)

            # Execute partial function to adjust current value of layer attr
            new_val = op_func(val)

            setattr(layer, attr, new_val)
    return model


def apply_filters(factor, cur_value, default_val=16):
    """
    Function that applied factor to the current value.

    Although, a default_val is provided to be possible to calculate the right new value.

    Imagine we have a convolution layer with filters = 16 and our factor is 2+3 = 5 because
    one 2 times and other 3 times added up.

    The problem is that the tree is processed recursively so if we multiply directly the factor with
    the current value would be something like: 16 * 2 * 3 = 96

    The result is wrong because we want: 16 * (2+3) = 80

    To circumvent that we always multiply the default value by the calculated factor and if
    the current value is not the default one (means that filters were already multiplied) just add
    the current value.

    In our example this would be: 16 * 2 = 32 then 16 * 3 + 32 = 80

    The default value for the filters is 16 because was said on the article.

    :param factor: Factor to be applied
    :param cur_value: Current value to multiplied by the factor
    :param default_val: Default value of filters number
    :return: New filters number value
    """
    new_val = default_val * factor

    if cur_value != default_val:
        new_val += cur_value

    return new_val


def apply_stride(factor, cur_value):
    """
    Function that applies the double to the current of strides in a StrideConv2D

    Isn't needed to accumulate like the apply_filters() function because is always doubling the current value

    :param factor: Normally value 2 to double the current value
    :param cur_value: Current value
    :return: New value for the stride attribute
    """
    # If strides is a list or tuple needs to be iterated to scale the strides value
    if isinstance(cur_value, (list, tuple)):
        new_val = [i * factor for i in cur_value]
    else:
        new_val = cur_value * factor

    return new_val
