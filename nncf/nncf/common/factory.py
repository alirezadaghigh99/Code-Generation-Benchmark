def create(model: TModel) -> NNCFGraph:
        """
        Factory method to create backend-specific NNCFGraph instance based on the input model.

        :param model: backend-specific model instance
        :return: backend-specific NNCFGraph instance
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.ONNX:
            from nncf.onnx.graph.nncf_graph_builder import GraphConverter

            return GraphConverter.create_nncf_graph(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.openvino.graph.nncf_graph_builder import GraphConverter

            return GraphConverter.create_nncf_graph(model)
        if model_backend == BackendType.TORCH:
            return model.nncf.get_graph()
        raise nncf.UnsupportedBackendError(
            "Cannot create backend-specific graph because {} is not supported!".format(model_backend.value)
        )

