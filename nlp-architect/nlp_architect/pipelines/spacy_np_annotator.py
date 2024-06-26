class SpacyNPAnnotator(object):
    """
    Simple Spacy pipe with NP extraction annotations
    """

    def __init__(self, model_path, settings_path, spacy_model="en", batch_size=32, use_cudnn=False):
        _model_path = path.join(path.dirname(path.realpath(__file__)), model_path)
        validate_existing_filepath(_model_path)
        _settings_path = path.join(path.dirname(path.realpath(__file__)), settings_path)
        validate_existing_filepath(_settings_path)

        nlp = spacy.load(spacy_model)
        for p in nlp.pipe_names:
            if p not in ["tagger"]:
                nlp.remove_pipe(p)
        nlp.add_pipe(nlp.create_pipe("sentencizer"), first=True)
        nlp.add_pipe(
            NPAnnotator.load(
                _model_path, settings_path, batch_size=batch_size, use_cudnn=use_cudnn
            ),
            last=True,
        )
        self.nlp = nlp

    def __call__(self, text: str) -> [str]:
        """
        Parse a given text and return a list of noun phrases found

        Args:
            text (str): a text string

        Returns:
            list of noun phrases as strings
        """
        return [np.text for np in get_noun_phrases(self.nlp(text))]