class EvaluateParser(JavaProtobufContext):
    """
    Parser evaluation context window

    This is a context window which keeps a process open.  Should allow
    for multiple requests without launching new java processes each time.
    """
    def __init__(self, classpath=None, kbest=None, silent=False):
        if kbest is not None:
            extra_args = ["-evalPCFGkBest", "{}".format(kbest), "-evals", "pcfgTopK"]
        else:
            extra_args = []

        if silent:
            extra_args.extend(["-evals", "summary=False"])

        super(EvaluateParser, self).__init__(classpath, EvaluateParserResponse, EVALUATE_JAVA, extra_args=extra_args)

    def process(self, treebank):
        request = build_request(treebank)
        return self.process_request(request)

