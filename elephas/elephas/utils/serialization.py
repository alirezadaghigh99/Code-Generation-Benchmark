def dict_to_model(_dict: dict, custom_objects: dict = None):
    """Turns a Python dictionary with model architecture and weights
    back into a Keras model

    :param _dict: dictionary with `model` and `weights` keys.
    :param custom_objects: custom objects i.e; layers/activations, required for model
    :return: Keras model instantiated from dictionary
    """
    model = model_from_json(_dict['model'], custom_objects)
    model.set_weights(_dict['weights'])
    return model