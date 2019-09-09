from enum import Enum, auto

import tensorflow as tf
import tensorflow.keras.layers as tf_layers


class ResidualType(Enum):
    ORIGINAL = auto()
    PRE_ACTIVATION = auto()


def _original_bottleneck_block(layer_factory, name, inner_filters, cardinality, out_filters,
                               stride, proj, last):
    res = last
    if proj:
        res = layer_factory.proj(name + "_res_prj", out_filters, stride)(res)
    last = layer_factory.conv2d(name + "_conv_r", inner_filters, 1, 1)(last)
    last = layer_factory.batch_norm(name + "_bn_r")(last)
    last = layer_factory.relu(name + "_relu_r")(last)
    last = layer_factory.conv2d(name + "_conv_i", inner_filters, 3, stride,
                                groups=cardinality)(last)
    last = layer_factory.batch_norm(name + "_bn_i")(last)
    last = layer_factory.relu(name + "_relu_i")(last)
    last = layer_factory.conv2d(name + "_conv_e", out_filters, 1, 1)(last)
    last = layer_factory.batch_norm(name + "_bn_e")(last)
    last = tf_layers.Add(name=name + "_res_add")([last, res])
    last = layer_factory.relu(name + "_relu_e")(last)
    return last


def _pre_activation_bottleneck_block(layer_factory, name, inner_filters, cardinality, out_filters,
                                     stride, proj, last):
    res = last
    last = layer_factory.batch_norm(name + "_bn_r")(last)
    last = layer_factory.relu(name + "_relu_r")(last)
    if proj:
        res = layer_factory.proj(name + "_res_prj", out_filters, stride)(last)
    last = layer_factory.conv2d(name + "_conv_r", inner_filters, 1, stride)(last)
    last = layer_factory.batch_norm(name + "_bn_i")(last)
    last = layer_factory.relu(name + "_relu_i")(last)
    last = layer_factory.conv2d(name + "_conv_i", inner_filters, 3, 1, groups=cardinality)(last)
    last = layer_factory.batch_norm(name + "_bn_e")(last)
    last = layer_factory.relu(name + "_relu_e")(last)
    last = layer_factory.conv2d(name + "_conv_e", out_filters, 1, 1)(last)
    last = tf_layers.Add(name=name + "_res_add")([last, res])
    return last


def resnext(
        layer_factory,
        cardinality,
        depth,
        base_width,
        residual_type=ResidualType.ORIGINAL,
        dim=64,
        widen_factor=4,
        num_blocks=3,
):
    if residual_type not in [ResidualType.ORIGINAL, ResidualType.PRE_ACTIVATION]:
        raise ValueError("Unknown residual_type: {}.".format(residual_type))
    if (depth - 2) % (3 * num_blocks) != 0:
        raise ValueError("depth - 2 must be divisible by 3 * num_blocks.")
    block_depth = (depth - 2) // (3 * num_blocks)
    width_ratios = [2**idx for idx in range(num_blocks)]
    num_filters = [dim] + [dim * widen_factor * wr for wr in width_ratios]
    input_tensor = tf.keras.Input(shape=[32, 32, 3], name="input")
    last = input_tensor
    last = tf_layers.Permute([3, 1, 2], name="transpose")(last)
    last = layer_factory.conv2d("conv", num_filters[0], 3, 1)(last)
    if residual_type == ResidualType.ORIGINAL:
        last = layer_factory.batch_norm("bn")(last)
        last = layer_factory.relu("relu")(last)
    for blk_idx in range(num_blocks):
        blk_name = "blk_{}".format(chr(ord("A") + blk_idx))
        inner_filters = cardinality * base_width * width_ratios[blk_idx]
        out_filters = num_filters[blk_idx + 1]
        for btlnk_idx in range(block_depth):
            name = "{}_btlnk_{}".format(blk_name, btlnk_idx)
            stride = 1 if (blk_idx == 0 or btlnk_idx > 0) else 2
            if residual_type == ResidualType.ORIGINAL:
                block_fn = _original_bottleneck_block
                proj = btlnk_idx == 0
            else:
                assert residual_type == ResidualType.PRE_ACTIVATION
                block_fn = _pre_activation_bottleneck_block
                proj = stride > 1 or (blk_idx == 0 and btlnk_idx == 0)
            last = block_fn(layer_factory, name, inner_filters, cardinality, out_filters, stride,
                            proj, last)
    if residual_type == ResidualType.PRE_ACTIVATION:
        last = layer_factory.batch_norm("batch_norm")(last)
        last = layer_factory.relu("relu")(last)
    last = layer_factory.avg_pooling("avg_pool")(last)
    last = layer_factory.dense("dense", units=10)(last)
    return tf.keras.Model(inputs=input_tensor, outputs=last)
