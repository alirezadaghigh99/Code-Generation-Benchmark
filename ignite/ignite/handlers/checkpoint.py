def load_objects(to_load: Mapping, checkpoint: Union[str, Mapping, Path], **kwargs: Any) -> None:
        """Helper method to apply ``load_state_dict`` on the objects from ``to_load`` using states from ``checkpoint``.

        Args:
            to_load: a dictionary with objects, e.g. `{"model": model, "optimizer": optimizer, ...}`
            checkpoint: a path, a string filepath or a dictionary with state_dicts to load, e.g.
                `{"model": model_state_dict, "optimizer": opt_state_dict}`. If `to_load` contains a single key,
                then checkpoint can contain directly corresponding state_dict.
            kwargs: Keyword arguments accepted for `nn.Module.load_state_dict()`. Passing `strict=False` enables
                the user to load part of the pretrained model (useful for example, in Transfer Learning)

        Examples:
            .. code-block:: python

                import tempfile
                from pathlib import Path

                import torch

                from ignite.engine import Engine, Events
                from ignite.handlers import ModelCheckpoint, Checkpoint

                trainer = Engine(lambda engine, batch: None)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    handler = ModelCheckpoint(tmpdirname, 'myprefix', n_saved=None, create_dir=True)

                    model = torch.nn.Linear(3, 3)
                    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

                    to_save = {"weights": model, "optimizer": optimizer}

                    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=2), handler, to_save)
                    trainer.run(torch.randn(10, 1), 5)

                    to_load = to_save
                    checkpoint_fp = Path(tmpdirname) / 'myprefix_checkpoint_40.pt'
                    checkpoint = torch.load(checkpoint_fp)
                    Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)

                    # or using a string for checkpoint filepath

                    to_load = to_save
                    checkpoint_fp = Path(tmpdirname) / 'myprefix_checkpoint_40.pt'
                    Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint_fp)

        Note:
            If ``to_load`` contains objects of type torch `DistributedDataParallel`_ or
            `DataParallel`_, method ``load_state_dict`` will applied to their internal wrapped model (``obj.module``).

        .. _DistributedDataParallel: https://pytorch.org/docs/stable/generated/
            torch.nn.parallel.DistributedDataParallel.html
        .. _DataParallel: https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
        """
        if not isinstance(checkpoint, (collections.Mapping, str, Path)):
            raise TypeError(f"Argument checkpoint should be a string or a dictionary, but given {type(checkpoint)}")

        Checkpoint._check_objects(to_load, "load_state_dict")

        if isinstance(checkpoint, (str, Path)):
            checkpoint_obj = torch.load(checkpoint)
        else:
            checkpoint_obj = checkpoint

        def _load_object(obj: Any, chkpt_obj: Any) -> None:
            if isinstance(obj, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                obj = obj.module

            if isinstance(obj, torch.nn.Module):
                obj.load_state_dict(chkpt_obj, **kwargs)
            else:
                obj.load_state_dict(chkpt_obj)

        if len(to_load) == 1:
            # single object and checkpoint is directly a state_dict
            key, obj = list(to_load.items())[0]
            if key not in checkpoint_obj:
                _load_object(obj, checkpoint_obj)
                return

        _tree_apply2(_load_object, to_load, checkpoint_obj)

