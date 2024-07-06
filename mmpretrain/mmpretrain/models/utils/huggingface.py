def build(**kwargs):
        if ('pretrained_model_name_or_path' not in kwargs
                and 'name_or_path' not in kwargs):
            raise TypeError(
                f'{cls.__name__} missing required argument '
                '`pretrained_model_name_or_path` or `name_or_path`.')
        # `pretrained_model_name_or_path` is too long for config,
        # add an alias name `name_or_path` here.
        name_or_path = kwargs.pop('pretrained_model_name_or_path',
                                  kwargs.pop('name_or_path'))

        if kwargs.pop('load_pretrained', True) and _load_hf_pretrained_model:
            model = cls.from_pretrained(name_or_path, **kwargs)
            setattr(model, 'is_init', True)
            return model
        else:
            cfg = get_config(name_or_path, **kwargs)
            return from_config(cfg)

def build(**kwargs):
        if ('pretrained_model_name_or_path' not in kwargs
                and 'name_or_path' not in kwargs):
            raise TypeError(
                f'{cls.__name__} missing required argument '
                '`pretrained_model_name_or_path` or `name_or_path`.')
        # `pretrained_model_name_or_path` is too long for config,
        # add an alias name `name_or_path` here.
        name_or_path = kwargs.pop('pretrained_model_name_or_path',
                                  kwargs.pop('name_or_path'))

        if kwargs.pop('load_pretrained', True) and _load_hf_pretrained_model:
            model = cls.from_pretrained(name_or_path, **kwargs)
            setattr(model, 'is_init', True)
            return model
        else:
            cfg = get_config(name_or_path, **kwargs)
            return from_config(cfg)

def build(**kwargs):
        if ('pretrained_model_name_or_path' not in kwargs
                and 'name_or_path' not in kwargs):
            raise TypeError(
                f'{cls.__name__} missing required argument '
                '`pretrained_model_name_or_path` or `name_or_path`.')
        # `pretrained_model_name_or_path` is too long for config,
        # add an alias name `name_or_path` here.
        name_or_path = kwargs.pop('pretrained_model_name_or_path',
                                  kwargs.pop('name_or_path'))

        if kwargs.pop('load_pretrained', True) and _load_hf_pretrained_model:
            model = cls.from_pretrained(name_or_path, **kwargs)
            setattr(model, 'is_init', True)
            return model
        else:
            cfg = get_config(name_or_path, **kwargs)
            return from_config(cfg)

