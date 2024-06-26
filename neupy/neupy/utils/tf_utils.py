def initialize_uninitialized_variables(variables=None):
    if variables is None:
        variables = tf.global_variables()

    if not variables:
        return

    session = tensorflow_session()
    is_not_initialized = session.run([
        tf.is_variable_initialized(var) for var in variables])

    not_initialized_vars = [
        v for (v, f) in zip(variables, is_not_initialized) if not f]

    if len(not_initialized_vars):
        session.run(tf.variables_initializer(not_initialized_vars))