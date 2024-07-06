def get(self, _id: ObjectId) -> DelegatedOperationDocument:
        """Get an operation by id."""
        raise NotImplementedError("subclass must implement get()")

