    def learn(self, data: pd.DataFrame) -> None:
        """
        Trains the attribute model starting from raw data. Raw data is pre-processed and cleaned accordingly. As data is assigned a particular type (ex: numerical, categorical, etc.), the respective feature encoder will convert it into a representation useable for training ML models. Of all ML models requested, these models are compiled and fit on the training data.

        This step amalgates ``preprocess`` -> ``featurize`` -> ``fit`` with the necessary splitting + analyze_data that occurs.

        :param data: (Unprocessed) Data used in training the model(s).

        :returns: Nothing; instantiates with best fit model from ensemble.
        """  # noqa
        pass