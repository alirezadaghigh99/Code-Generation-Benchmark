class KIEDocument(Document):
    """Implements a document element as a collection of pages

    Args:
    ----
        pages: list of page elements
    """

    _children_names: List[str] = ["pages"]
    pages: List[KIEPage] = []  # type: ignore[assignment]

    def __init__(
        self,
        pages: List[KIEPage],
    ) -> None:
        super().__init__(pages=pages)  # type: ignore[arg-type]