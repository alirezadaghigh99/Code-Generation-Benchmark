class Tsurgeon(JavaProtobufContext):
    """
    Tsurgeon context window

    This is a context window which keeps a process open.  Should allow
    for multiple requests without launching new java processes each time.
    """
    def __init__(self, classpath=None):
        super(Tsurgeon, self).__init__(classpath, TsurgeonResponse, TSURGEON_JAVA)

    def process(self, trees, *operations):
        request = build_request(trees, operations)
        result = self.process_request(request)
        return [from_tree(t)[0] for t in result.trees]

