def barrier(self, opts=None):
        store = c10d._get_default_store()
        key = "TEST:DummyProcessGroup:barrier"
        if self.rank() == 0:
            worker_count = 0
            # By default, TCPServer lives on rank 0. So rank 0 needs to make
            # sure that it does not exit too early before other ranks finish
            # using the store.
            # Note that, _store_based_barrier does not solve this problem, as
            # all ranks need to run at least one store.add(key, 0) before
            # exiting, but there is no guarantee that rank 0 is still alive at
            # that point.
            while worker_count < self.size() - 1:
                worker_count = store.add(key, 0)
        else:
            store.add(key, 1)

        return DummyWork()

