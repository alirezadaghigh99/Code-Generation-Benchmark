    def from_networkx(cls, graph) -> "Graph":
        """
        Creates a new Graph instance from a networkx graph.
        """
        ig = igraph.Graph.from_networkx(graph)
        return cls.from_igraph(ig)