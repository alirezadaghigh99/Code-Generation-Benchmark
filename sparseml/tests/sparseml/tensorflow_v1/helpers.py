def mlp_net():
    inp = tf_compat.placeholder(tf_compat.float32, [None, 16], name="inp")

    with tf_compat.name_scope("mlp_net"):
        fc1 = _fc("fc1", inp, 16, 32)
        fc2 = _fc("fc2", fc1, 32, 64)
        fc3 = _fc("fc3", fc2, 64, 64, add_relu=False)

    out = tf_compat.sigmoid(fc3, name="out")

    return out, inp

