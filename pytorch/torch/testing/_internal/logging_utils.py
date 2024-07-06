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

def make_logging_test(**kwargs):
    def wrapper(fn):
        @inductor_config.patch({"fx_graph_cache": False})
        def test_fn(self):

            torch._dynamo.reset()
            records = []
            # run with env var
            if len(kwargs) == 0:
                with self._handler_watcher(records):
                    fn(self, records)
            else:
                with log_settings(kwargs_to_settings(**kwargs)), self._handler_watcher(records):
                    fn(self, records)

            # run with API
            torch._dynamo.reset()
            records.clear()
            with log_api(**kwargs), self._handler_watcher(records):
                fn(self, records)


        return test_fn

    return wrapper

