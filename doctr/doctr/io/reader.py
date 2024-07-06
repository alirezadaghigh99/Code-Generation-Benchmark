def from_pdf(cls, file: AbstractFile, **kwargs) -> List[np.ndarray]:
        """Read a PDF file

        >>> from doctr.io import DocumentFile
        >>> doc = DocumentFile.from_pdf("path/to/your/doc.pdf")

        Args:
        ----
            file: the path to the PDF file or a binary stream
            **kwargs: additional parameters to :meth:`pypdfium2.PdfPage.render`

        Returns:
        -------
            the list of pages decoded as numpy ndarray of shape H x W x 3
        """
        return read_pdf(file, **kwargs)

