class CategoryMismatchTrainTest(NewCategoryTrainTest):
    """Find new categories in the test set."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn('CategoryMismatchTrainTest is deprecated, use NewCategoryTrainTest instead',
                      DeprecationWarning)

    pass

