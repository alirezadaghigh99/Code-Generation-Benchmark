def preprocess(request: Dict):
    redis_client = Redis(connection_pool=pool)
    with failure_handler(redis_client, request["id"]):
        model_manager.add_model(request["model_id"], request["api_key"])
        model_type = model_manager.get_task_type(request["model_id"])
        request = request_from_type(model_type, request)
        image, preprocess_return_metadata = model_manager.preprocess(
            request.model_id, request
        )
        # multi image requests are split into single image requests upstream and rebatched later
        image = image[0]
        request.image.value = None  # avoid writing image again since it's in memory
        shm = shared_memory.SharedMemory(create=True, size=image.nbytes)
        with shm_manager(shm):
            shared = np.ndarray(image.shape, dtype=image.dtype, buffer=shm.buf)
            shared[:] = image[:]
            shm_metadata = SharedMemoryMetadata(shm.name, image.shape, image.dtype.name)
            queue_infer_task(
                redis_client, shm_metadata, request, preprocess_return_metadata
            )