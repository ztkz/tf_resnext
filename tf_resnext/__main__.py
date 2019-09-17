import tensorflow as tf
from absl import app

from tf_resnext import data_utils
from tf_resnext import metrics
from tf_resnext import models
from tf_resnext.flags import FLAGS
from tf_resnext.layers import LayerFactory


def main(argv):
    del argv  # Unused.

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.per_process_gpu_memory_fraction
    tf.keras.backend.set_session(tf.Session(config=config))

    (train, train_steps), (test, test_steps) = data_utils.load_cifar10_data()
    train, train_steps = data_utils.preprocess_dataset(
        train,
        train_steps,
        FLAGS.batch_size,
        map_fns=[data_utils.random_pad_and_crop, data_utils.random_flip_left_right],
    )
    test, test_steps = data_utils.preprocess_dataset(test, test_steps, FLAGS.batch_size)

    layer_factory = LayerFactory(weight_decay=FLAGS.weight_decay)
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = models.resnext(
            layer_factory=layer_factory,
            cardinality=FLAGS.cardinality,
            depth=FLAGS.depth,
            base_width=FLAGS.base_width,
            residual_type=models.ResidualType[FLAGS.residual_type],
        )
        model.compile(
            loss=metrics.sparse_cross_entropy_loss,
            optimizer=tf.keras.optimizers.SGD(lr=0.0,
                                              momentum=FLAGS.momentum,
                                              nesterov=FLAGS.use_nesterov),
            metrics=[metrics.accuracy],
        )
        assert model.built

    model.summary()

    best_test_accuracy = 0.0
    next_lr_val = FLAGS.learning_rate
    for epochs in map(int, FLAGS.training_schedule):
        tf.keras.backend.set_value(model.optimizer.lr, next_lr_val)
        next_lr_val *= FLAGS.learning_rate_decay_factor
        history = model.fit(
            train,
            epochs=epochs,
            validation_data=test,
            steps_per_epoch=train_steps,
            validation_steps=test_steps,
        )
        best_test_accuracy = max(best_test_accuracy, max(history.history["val_accuracy"]))
        print("Best testing set accuracy: {:.4f}.".format(best_test_accuracy))


def run_app():
    app.run(main)


if __name__ == "__main__":
    run_app()
