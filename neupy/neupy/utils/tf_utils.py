def tensorflow_session():
    if hasattr(tensorflow_session, 'cache'):
        session = tensorflow_session.cache

        if not session._closed:
            return session

    config = tf.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
    )
    session = tf.Session(config=config)

    tensorflow_session.cache = session
    return session

