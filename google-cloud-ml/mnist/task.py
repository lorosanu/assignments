import tensorflow as tf

from . import model, utils

def train_and_evaluate(hparams):
    """Run the training and evaluation of the model"""

    # load data
    (train_images, train_labels), (test_images, test_labels) = \
        utils.load_mnist_data(hparams.local)

    # create estimator
    estimator = model.build_estimator(hparams.job_dir, hparams.shallow)

    # create specs
    train_spec = tf.estimator.TrainSpec(
        lambda: utils.feed_data(
            train_images,
            train_labels,
            batch_size=hparams.train_batch_size),
        max_steps=hparams.train_steps)

    exporter = tf.estimator.FinalExporter('mnist', utils.serving_input)
    eval_spec = tf.estimator.EvalSpec(
        lambda: utils.feed_data(
            test_images,
            test_labels,
            batch_size=hparams.eval_batch_size,
            shuffle=False),
        steps=hparams.eval_steps,
        exporters=[exporter],
        name='mnist-eval')

    # train and evaluate
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == '__main__':
    args = utils.get_arguments()
    tf.logging.set_verbosity(args.verbosity)
    train_and_evaluate(args)
