    def create_from_matrix(confusion_matrix):
        assert confusion_matrix.shape[0] == confusion_matrix.shape[1]
        matrix = ConfusionMatrix(confusion_matrix.shape[0])
        matrix.confusion_matrix = confusion_matrix
        return matrix