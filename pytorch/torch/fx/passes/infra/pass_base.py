class PassResult(namedtuple("PassResult", ["graph_module", "modified"])):
    """
    Result of a pass:
        graph_module: The modified graph module
        modified: A flag for if the pass has modified the graph module
    """
    def __new__(cls, graph_module, modified):
        return super().__new__(cls, graph_module, modified)

