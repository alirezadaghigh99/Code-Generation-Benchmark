class SparkModel(object):

    def __init__(self, model, mode='asynchronous', frequency='epoch', parameter_server_mode='http', num_workers=None,
                 custom_objects=None, batch_size=32, port=4000, *args, **kwargs):
        """SparkModel

        Base class for distributed training on RDDs. Spark model takes a Keras
        model as master network, an optimization scheme, a parallelisation mode
        and an averaging frequency.

        :param model: Compiled Keras model
        :param mode: String, choose from `asynchronous`, `synchronous` and `hogwild`
        :param frequency: String, either `epoch` or `batch`
        :param parameter_server_mode: String, either `http` or `socket`
        :param num_workers: int, number of workers used for training (defaults to None)
        :param custom_objects: Keras custom objects
        :param batch_size: batch size used for training and inference
        :param port: port used in case of 'http' parameter server mode
        """
        self._training_histories = []
        self._master_network = model
        if not hasattr(model, "loss"):
            raise Exception(
                "Compile your Keras model before initializing an Elephas model with it")
        metrics = model.compiled_metrics._metrics
        loss = model.loss
        optimizer = serialize_optimizer(model.optimizer)

        if custom_objects is None:
            custom_objects = {}
        if metrics is None:
            metrics = ["accuracy"]
        self.mode = mode
        self.frequency = frequency
        self.num_workers = num_workers
        self.weights = self._master_network.get_weights()
        self.pickled_weights = None
        self.master_optimizer = optimizer
        self.master_loss = loss
        self.master_metrics = metrics
        self.custom_objects = custom_objects
        self.parameter_server_mode = parameter_server_mode
        self.batch_size = batch_size
        self.port = port
        self.kwargs = kwargs

        self.serialized_model = model_to_dict(model)
        if self.mode is not 'synchronous':
            factory = ClientServerFactory.get_factory(self.parameter_server_mode)
            self.parameter_server = factory.create_server(self.serialized_model, self.port, self.mode,
                                                          custom_objects=self.custom_objects)
            self.client = factory.create_client(self.port)

    def get_config(self):
        base_config = {
            'parameter_server_mode': self.parameter_server_mode,
            'mode': self.mode,
            'frequency': self.frequency,
            'num_workers': self.num_workers,
            'batch_size': self.batch_size}
        config = base_config.copy()
        config.update(self.kwargs)
        return config

    def save(self, file_name):
        model = self._master_network
        model.save(file_name)
        f = h5py.File(file_name, mode='a')

        f.attrs['distributed_config'] = json.dumps({
            'class_name': self.__class__.__name__,
            'config': self.get_config()
        }).encode('utf8')

        f.flush()
        f.close()

    @property
    def training_histories(self):
        return self._training_histories

    @property
    def master_network(self):
        return self._master_network

    @master_network.setter
    def master_network(self, network):
        self._master_network = network

    def start_server(self):
        self.parameter_server.start()

    def stop_server(self):
        self.parameter_server.stop()

    def predict(self, data: Union[RDD, np.array]):
        """Get prediction probabilities for a numpy array of features
        """
        if isinstance(data, (np.ndarray,)):
            from pyspark.sql import SparkSession
            sc = SparkSession.builder.getOrCreate().sparkContext
            data = sc.parallelize(data)
        return self._predict(data)

    def evaluate(self, x_test, y_test, **kwargs):
        from pyspark.sql import SparkSession
        sc = SparkSession.builder.getOrCreate().sparkContext
        test_rdd = to_simple_rdd(sc, x_test, y_test)
        return self._evaluate(test_rdd, **kwargs)

    def fit(self, rdd: RDD, **kwargs):
        """
        Train an elephas model on an RDD. The Keras model configuration as specified
        in the elephas model is sent to Spark workers, abd each worker will be trained
        on their data partition.

        :param rdd: RDD with features and labels
        :param epochs: number of epochs used for training
        :param batch_size: batch size used for training
        :param verbose: logging verbosity level (0, 1 or 2)
        :param validation_split: percentage of data set aside for validation
        """
        print('>>> Fit model')
        if self.num_workers:
            rdd = rdd.repartition(self.num_workers)

        if self.mode in ['asynchronous', 'synchronous', 'hogwild']:
            self._fit(rdd, **kwargs)
        else:
            raise ValueError(
                "Choose from one of the modes: asynchronous, synchronous or hogwild")

    def _fit(self, rdd: RDD, **kwargs):
        """Protected train method to make wrapping of modes easier
        """
        self._master_network.compile(optimizer=get_optimizer(self.master_optimizer),
                                     loss=self.master_loss,
                                     metrics=self.master_metrics)
        if self.mode in ['asynchronous', 'hogwild']:
            self.start_server()
        train_config = kwargs
        freq = self.frequency
        optimizer = deserialize_optimizer(self.master_optimizer)
        loss = self.master_loss
        metrics = self.master_metrics
        custom = self.custom_objects

        model_json = self._master_network.to_json()
        init = self._master_network.get_weights()
        parameters = rdd.context.broadcast(init)

        if self.mode in ['asynchronous', 'hogwild']:
            print('>>> Initialize workers')
            worker = AsynchronousSparkWorker(
                model_json, parameters, self.client, train_config, freq, optimizer, loss, metrics, custom)
            print('>>> Distribute load')
            rdd.mapPartitions(worker.train).collect()
            print('>>> Async training complete.')
            new_parameters = self.client.get_parameters()
        elif self.mode == 'synchronous':
            worker = SparkWorker(model_json, parameters, train_config,
                                 optimizer, loss, metrics, custom)
            training_outcomes = rdd.mapPartitions(worker.train).collect()
            new_parameters = self._master_network.get_weights()
            number_of_sub_models = len(training_outcomes)
            for training_outcome in training_outcomes:
                grad, history = training_outcome
                self.training_histories.append(history)
                weighted_grad = divide_by(grad, number_of_sub_models)
                new_parameters = subtract_params(new_parameters, weighted_grad)
            print('>>> Synchronous training complete.')
        else:
            raise ValueError("Unsupported mode {}".format(self.mode))
        self._master_network.set_weights(new_parameters)
        if self.mode in ['asynchronous', 'hogwild']:
            self.stop_server()

    def _predict(self, rdd: RDD):
        json_model = self.master_network.to_json()
        weights = self.master_network.get_weights()
        weights = rdd.context.broadcast(weights)
        custom_objs = self.custom_objects

        def _predict(model_as_json, custom_objects, data):
            model = model_from_json(model_as_json, custom_objects)
            model.set_weights(weights.value)
            data = np.array([x for x in data])
            return model.predict(data)

        def _predict_with_indices(model_as_json, custom_objects, data):
            model = model_from_json(model_as_json, custom_objects)
            model.set_weights(weights.value)
            data, indices = zip(*data)
            data = np.array(data)
            return zip(model.predict(data), indices)

        if self.num_workers and self.num_workers > 1:
            # if there are multiple workers, we need to retrieve element indices and preserve them throughout
            # the inference process, since we'll need to sort by index before returning the result, as repartitioning
            # does not preserve ordering, but the users will expect prediction results which correspond to the ordering
            # of samples they supplied.
            rdd = rdd.zipWithIndex()
            rdd = rdd.repartition(self.num_workers)
            predictions_and_indices = rdd.mapPartitions(partial(_predict_with_indices, json_model, custom_objs))
            predictions_sorted_by_index = predictions_and_indices.sortBy(lambda x: x[1])
            predictions = predictions_sorted_by_index.map(lambda x: x[0]).collect()
        else:
            # if there are no workers specified or only a single worker, we don't need to worry about handling index
            # values, since there will be no shuffling
            predictions = rdd.mapPartitions(partial(_predict, json_model, custom_objs)).collect()

        return predictions

    def _evaluate(self, rdd: RDD, **kwargs):
        json_model = self.master_network.to_json()
        optimizer = deserialize_optimizer(self.master_optimizer)
        loss = self.master_loss
        weights = self.master_network.get_weights()
        weights = rdd.context.broadcast(weights)
        custom_objects = self.custom_objects
        metrics = self.master_metrics

        def _evaluate(model, optimizer, loss, custom_objects, metrics, kwargs, data_iterator):
            model = model_from_json(model, custom_objects)
            model.compile(optimizer, loss, metrics)
            model.set_weights(weights.value)
            feature_iterator, label_iterator = tee(data_iterator, 2)
            x_test = np.asarray([x for x, y in feature_iterator])
            y_test = np.asarray([y for x, y in label_iterator])
            return [model.evaluate(x_test, y_test, **kwargs)]

        if self.num_workers:
            rdd = rdd.repartition(self.num_workers)
        results = rdd.mapPartitions(partial(_evaluate, json_model, optimizer, loss, custom_objects, metrics, kwargs))
        if not metrics:
            # if no metrics, we can just return the scalar corresponding to the loss value
            return results.mean()
        else:
            # if we do have metrics, we want to return a list of [loss value, metric value] - to match the keras API
            loss_value = results.map(lambda x: x[0]).mean()
            metric_value = results.map(lambda x: x[1]).mean()
            return [loss_value, metric_value]

