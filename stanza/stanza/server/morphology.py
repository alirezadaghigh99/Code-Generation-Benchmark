class Morphology(JavaProtobufContext):
    """
    Morphology context window

    This is a context window which keeps a process open.  Should allow
    for multiple requests without launching new java processes each time.

    (much faster than calling process_text over and over)
    """
    def __init__(self, classpath=None):
        super(Morphology, self).__init__(classpath, MorphologyResponse, MORPHOLOGY_JAVA)

    def process(self, words, xpos_tags):
        """
        Get the lemmata for each word/tag pair
        """
        request = build_request(words, xpos_tags)
        return self.process_request(request)

