# TensorFlow ResNeXt

TensorFlow reimplementation of [ResNeXt](https://arxiv.org/abs/1611.05431) for CIFAR-10.

# Running

Run it like this:

```
pip install .
tf_resnext
```

or after installing the [dependencies](#dependencies) manually:

```
python -m tf_resnext
```

This should give ~96.4% best accuracy on the test set.

# Dependencies

```
absl-py
gast==0.2.2
numpy
tensorflow-gpu==1.14.0
```
