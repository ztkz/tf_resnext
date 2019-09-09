from functools import partial

import tensorflow as tf
import tensorflow.keras.layers as layers


class GroupedConv(tf.keras.Model):
    def __init__(self, filters, kernel_size, groups=1, **kwargs):
        super(GroupedConv, self).__init__(name=kwargs.get("name", None))
        if groups == 1:
            self._convs = [layers.Conv2D(filters, kernel_size, **kwargs)]
            self._conc = lambda x: x[0]
            return

        assert filters % groups == 0
        filters //= groups
        assert kwargs.get("data_format", "channels_first") == "channels_first"
        conc_axis = -3

        def slice_fn(group, input_tensor):
            channels = input_tensor.shape[1]
            assert channels % groups == 0
            step = channels // groups
            return input_tensor[:, group * step:(group + 1) * step, :, :]

        self._convs = [
            tf.keras.Sequential([
                layers.Lambda(partial(slice_fn, group)),
                layers.Conv2D(filters, kernel_size, **kwargs),
            ]) for group in range(groups)
        ]

        self._conc = layers.Concatenate(axis=conc_axis)

    def call(self, input_tensor):
        return self._conc([conv(input_tensor) for conv in self._convs])


class LayerFactory(object):
    def __init__(self, weight_decay):
        self._dense_initializer = tf.keras.initializers.he_normal()
        self._conv_initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_out")
        self._regularizer = tf.keras.regularizers.l2(weight_decay)

    def conv2d(self, name, filters, kernel_size, strides, **kwargs):
        kwargs.setdefault("padding", "same")
        kwargs.setdefault("use_bias", False)
        kwargs.setdefault("kernel_initializer", self._conv_initializer)
        kwargs.setdefault("kernel_regularizer", self._regularizer)
        kwargs.setdefault("data_format", "channels_first")
        return GroupedConv(filters=filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           name=name,
                           **kwargs)

    def batch_norm(self, name, **kwargs):
        kwargs.setdefault("axis", -3)
        kwargs.setdefault("trainable", True)
        kwargs.setdefault("momentum", 0.9)
        kwargs.setdefault("epsilon", 1e-5)
        kwargs.setdefault("dtype", tf.float32)
        kwargs.setdefault("beta_regularizer", self._regularizer)
        kwargs.setdefault("gamma_regularizer", self._regularizer)
        return layers.BatchNormalization(name=name, **kwargs)

    def dense(self, name, **kwargs):
        kwargs.setdefault("kernel_initializer", self._dense_initializer)
        kwargs.setdefault("kernel_regularizer", self._regularizer)
        kwargs.setdefault("bias_initializer", "zeros")
        kwargs.setdefault("bias_regularizer", self._regularizer)
        return layers.Dense(name=name, **kwargs)

    def avg_pooling(self, name, **kwargs):
        kwargs.setdefault("data_format", "channels_first")
        return layers.GlobalAveragePooling2D(name=name, **kwargs)

    def relu(self, name, **kwargs):
        return layers.Activation("relu", name=name, **kwargs)

    def proj(self, name, filters, strides):
        conv = self.conv2d(name + "_conv", filters, 1, strides)
        bn = self.batch_norm(name + "_bn")
        return lambda input_tensor: bn(conv(input_tensor))
