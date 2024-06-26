class InferencePipelinesManagerHandler(BaseRequestHandler):
    def __init__(
        self,
        request: socket.socket,
        client_address: Any,
        server: BaseServer,
        processes_table: Dict[str, Tuple[Process, Queue, Queue]],
    ):
        self._processes_table = processes_table  # in this case it's required to set the state of class before superclass init - as it invokes handle()
        super().__init__(request, client_address, server)

    def handle(self) -> None:
        pipeline_id: Optional[str] = None
        request_id = str(uuid4())
        try:
            data = receive_socket_data(
                source=self.request,
                header_size=HEADER_SIZE,
                buffer_size=SOCKET_BUFFER_SIZE,
            )
            data[TYPE_KEY] = CommandType(data[TYPE_KEY])
            if data[TYPE_KEY] is CommandType.LIST_PIPELINES:
                return self._list_pipelines(request_id=request_id)
            if data[TYPE_KEY] is CommandType.INIT:
                return self._initialise_pipeline(request_id=request_id, command=data)
            pipeline_id = data[PIPELINE_ID_KEY]
            if data[TYPE_KEY] is CommandType.TERMINATE:
                self._terminate_pipeline(
                    request_id=request_id, pipeline_id=pipeline_id, command=data
                )
            else:
                response = handle_command(
                    processes_table=self._processes_table,
                    request_id=request_id,
                    pipeline_id=pipeline_id,
                    command=data,
                )
                serialised_response = prepare_response(
                    request_id=request_id, response=response, pipeline_id=pipeline_id
                )
                send_data_trough_socket(
                    target=self.request,
                    header_size=HEADER_SIZE,
                    data=serialised_response,
                    request_id=request_id,
                    pipeline_id=pipeline_id,
                )
        except (KeyError, ValueError, MalformedPayloadError) as error:
            logger.error(
                f"Invalid payload in processes manager. error={error} request_id={request_id}..."
            )
            payload = prepare_error_response(
                request_id=request_id,
                error=error,
                error_type=ErrorType.INVALID_PAYLOAD,
                pipeline_id=pipeline_id,
            )
            send_data_trough_socket(
                target=self.request,
                header_size=HEADER_SIZE,
                data=payload,
                request_id=request_id,
                pipeline_id=pipeline_id,
            )
        except Exception as error:
            logger.error(
                f"Internal error in processes manager. error={error} request_id={request_id}..."
            )
            payload = prepare_error_response(
                request_id=request_id,
                error=error,
                error_type=ErrorType.INTERNAL_ERROR,
                pipeline_id=pipeline_id,
            )
            send_data_trough_socket(
                target=self.request,
                header_size=HEADER_SIZE,
                data=payload,
                request_id=request_id,
                pipeline_id=pipeline_id,
            )

    def _list_pipelines(self, request_id: str) -> None:
        serialised_response = prepare_response(
            request_id=request_id,
            response={
                "pipelines": list(self._processes_table.keys()),
                STATUS_KEY: OperationStatus.SUCCESS,
            },
            pipeline_id=None,
        )
        send_data_trough_socket(
            target=self.request,
            header_size=HEADER_SIZE,
            data=serialised_response,
            request_id=request_id,
        )

    def _initialise_pipeline(self, request_id: str, command: dict) -> None:
        pipeline_id = str(uuid4())
        command_queue = Queue()
        responses_queue = Queue()
        inference_pipeline_manager = InferencePipelineManager.init(
            command_queue=command_queue,
            responses_queue=responses_queue,
        )
        inference_pipeline_manager.start()
        self._processes_table[pipeline_id] = (
            inference_pipeline_manager,
            command_queue,
            responses_queue,
        )
        command_queue.put((request_id, command))
        response = get_response_ignoring_thrash(
            responses_queue=responses_queue, matching_request_id=request_id
        )
        serialised_response = prepare_response(
            request_id=request_id, response=response, pipeline_id=pipeline_id
        )
        send_data_trough_socket(
            target=self.request,
            header_size=HEADER_SIZE,
            data=serialised_response,
            request_id=request_id,
            pipeline_id=pipeline_id,
        )

    def _terminate_pipeline(
        self, request_id: str, pipeline_id: str, command: dict
    ) -> None:
        response = handle_command(
            processes_table=self._processes_table,
            request_id=request_id,
            pipeline_id=pipeline_id,
            command=command,
        )
        if response[STATUS_KEY] is OperationStatus.SUCCESS:
            logger.info(
                f"Joining inference pipeline. pipeline_id={pipeline_id} request_id={request_id}"
            )
            join_inference_pipeline(
                processes_table=self._processes_table, pipeline_id=pipeline_id
            )
            logger.info(
                f"Joined inference pipeline. pipeline_id={pipeline_id} request_id={request_id}"
            )
        serialised_response = prepare_response(
            request_id=request_id, response=response, pipeline_id=pipeline_id
        )
        send_data_trough_socket(
            target=self.request,
            header_size=HEADER_SIZE,
            data=serialised_response,
            request_id=request_id,
            pipeline_id=pipeline_id,
        )