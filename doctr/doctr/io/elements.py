class KIEPage(Element):
    """Implements a KIE page element as a collection of predictions

    Args:
    ----
        predictions: Dictionary with list of block elements for each detection class
        page: image encoded as a numpy array in uint8
        page_idx: the index of the page in the input raw document
        dimensions: the page size in pixels in format (height, width)
        orientation: a dictionary with the value of the rotation angle in degress and confidence of the prediction
        language: a dictionary with the language value and confidence of the prediction
    """

    _exported_keys: List[str] = ["page_idx", "dimensions", "orientation", "language"]
    _children_names: List[str] = ["predictions"]
    predictions: Dict[str, List[Prediction]] = {}

    def __init__(
        self,
        page: np.ndarray,
        predictions: Dict[str, List[Prediction]],
        page_idx: int,
        dimensions: Tuple[int, int],
        orientation: Optional[Dict[str, Any]] = None,
        language: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(predictions=predictions)
        self.page = page
        self.page_idx = page_idx
        self.dimensions = dimensions
        self.orientation = orientation if isinstance(orientation, dict) else dict(value=None, confidence=None)
        self.language = language if isinstance(language, dict) else dict(value=None, confidence=None)

    def render(self, prediction_break: str = "\n\n") -> str:
        """Renders the full text of the element"""
        return prediction_break.join(
            f"{class_name}: {p.render()}" for class_name, predictions in self.predictions.items() for p in predictions
        )

    def extra_repr(self) -> str:
        return f"dimensions={self.dimensions}"

    def show(self, interactive: bool = True, preserve_aspect_ratio: bool = False, **kwargs) -> None:
        """Overlay the result on a given image

        Args:
            interactive: whether the display should be interactive
            preserve_aspect_ratio: pass True if you passed True to the predictor
            **kwargs: keyword arguments passed to the matplotlib.pyplot.show method
        """
        requires_package("matplotlib", "`.show()` requires matplotlib & mplcursors installed")
        requires_package("mplcursors", "`.show()` requires matplotlib & mplcursors installed")
        import matplotlib.pyplot as plt

        visualize_kie_page(
            self.export(), self.page, interactive=interactive, preserve_aspect_ratio=preserve_aspect_ratio
        )
        plt.show(**kwargs)

    def synthesize(self, **kwargs) -> np.ndarray:
        """Synthesize the page from the predictions

        Args:
        ----
            **kwargs: keyword arguments passed to the matplotlib.pyplot.show method

        Returns:
        -------
            synthesized page
        """
        return synthesize_kie_page(self.export(), **kwargs)

    def export_as_xml(self, file_title: str = "docTR - XML export (hOCR)") -> Tuple[bytes, ET.ElementTree]:
        """Export the page as XML (hOCR-format)
        convention: https://github.com/kba/hocr-spec/blob/master/1.2/spec.md

        Args:
        ----
            file_title: the title of the XML file

        Returns:
        -------
            a tuple of the XML byte string, and its ElementTree
        """
        p_idx = self.page_idx
        prediction_count: int = 1
        height, width = self.dimensions
        language = self.language if "language" in self.language.keys() else "en"
        # Create the XML root element
        page_hocr = ETElement("html", attrib={"xmlns": "http://www.w3.org/1999/xhtml", "xml:lang": str(language)})
        # Create the header / SubElements of the root element
        head = SubElement(page_hocr, "head")
        SubElement(head, "title").text = file_title
        SubElement(head, "meta", attrib={"http-equiv": "Content-Type", "content": "text/html; charset=utf-8"})
        SubElement(
            head,
            "meta",
            attrib={"name": "ocr-system", "content": f"python-doctr {doctr.__version__}"},  # type: ignore[attr-defined]
        )
        SubElement(
            head,
            "meta",
            attrib={"name": "ocr-capabilities", "content": "ocr_page ocr_carea ocr_par ocr_line ocrx_word"},
        )
        # Create the body
        body = SubElement(page_hocr, "body")
        SubElement(
            body,
            "div",
            attrib={
                "class": "ocr_page",
                "id": f"page_{p_idx + 1}",
                "title": f"image; bbox 0 0 {width} {height}; ppageno 0",
            },
        )
        # iterate over the blocks / lines / words and create the XML elements in body line by line with the attributes
        for class_name, predictions in self.predictions.items():
            for prediction in predictions:
                if len(prediction.geometry) != 2:
                    raise TypeError("XML export is only available for straight bounding boxes for now.")
                (xmin, ymin), (xmax, ymax) = prediction.geometry
                prediction_div = SubElement(
                    body,
                    "div",
                    attrib={
                        "class": "ocr_carea",
                        "id": f"{class_name}_prediction_{prediction_count}",
                        "title": f"bbox {int(round(xmin * width))} {int(round(ymin * height))} \
                        {int(round(xmax * width))} {int(round(ymax * height))}",
                    },
                )
                prediction_div.text = prediction.value
                prediction_count += 1

        return ET.tostring(page_hocr, encoding="utf-8", method="xml"), ET.ElementTree(page_hocr)

    @classmethod
    def from_dict(cls, save_dict: Dict[str, Any], **kwargs):
        kwargs = {k: save_dict[k] for k in cls._exported_keys}
        kwargs.update({
            "predictions": [Prediction.from_dict(predictions_dict) for predictions_dict in save_dict["predictions"]]
        })
        return cls(**kwargs)