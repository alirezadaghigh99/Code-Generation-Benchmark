def make_settings_test(settings):
    def wrapper(fn):
        def test_fn(self):
            torch._dynamo.reset()
            records = []
            # run with env var
            with log_settings(settings), self._handler_watcher(records):
                fn(self, records)

        return test_fn

    return wrapper