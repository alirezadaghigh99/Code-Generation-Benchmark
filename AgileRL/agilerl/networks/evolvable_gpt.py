    def remove_node(self, numb_new_nodes=None):
        """Removes nodes from hidden layers of transformer.

        :param numb_new_nodes: Number of nodes to remove from hidden layers, defaults to None
        :type numb_new_nodes: int, optional
        """
        if numb_new_nodes is None:
            numb_new_nodes = np.random.choice([32, 64, 128], 1)[0]
        self.dim_feedfwd -= numb_new_nodes
        self.recreate_shrunk_nets()
        return {"numb_new_nodes": numb_new_nodes}