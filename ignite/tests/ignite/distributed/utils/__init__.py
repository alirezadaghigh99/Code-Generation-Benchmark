def _test_sync(cls):
    from ignite.distributed.utils import _SerialModel, _set_model

    _set_model(_SerialModel())

    sync()

    from ignite.distributed.utils import _model

    assert isinstance(_model, cls), f"{type(_model)} vs {cls}"

