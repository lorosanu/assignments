import tensorflow as tf

def build_estimator(job_dir, shallow=True, learning_rate=0.01):
    """Create the estimator instance for training and evaluation"""

    # build model
    if shallow:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(512, input_shape=(784,), activation='relu'),
            tf.keras.layers.Dropout(0.6),
            tf.keras.layers.Dense(10, activation='softmax')])
    else:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Reshape((28, 28, 1), input_shape=(784,)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='softmax')])

    # compile model
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    # create estimator
    run_config = tf.estimator.RunConfig()
    estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model,
        model_dir=job_dir,
        config=run_config)

    return estimator
