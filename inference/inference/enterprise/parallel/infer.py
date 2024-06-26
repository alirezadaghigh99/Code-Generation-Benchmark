    def infer(self):
        while True:
            model_id, images, batch, preproc_return_metadatas = self.batch_queue.get()
            outputs = self.model_manager.predict(model_id, images)
            for output, b, metadata in zip(
                zip(*outputs), batch, preproc_return_metadatas
            ):
                self.response_queue.put_nowait((output, b["request"], metadata))