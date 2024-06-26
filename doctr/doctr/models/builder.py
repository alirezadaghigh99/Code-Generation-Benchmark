class KIEDocumentBuilder(DocumentBuilder):
    """Implements a KIE document builder

    Args:
    ----
        resolve_lines: whether words should be automatically grouped into lines
        resolve_blocks: whether lines should be automatically grouped into blocks
        paragraph_break: relative length of the minimum space separating paragraphs
        export_as_straight_boxes: if True, force straight boxes in the export (fit a rectangle
            box to all rotated boxes). Else, keep the boxes format unchanged, no matter what it is.
    """

    def __call__(  # type: ignore[override]
        self,
        pages: List[np.ndarray],
        boxes: List[Dict[str, np.ndarray]],
        objectness_scores: List[Dict[str, np.ndarray]],
        text_preds: List[Dict[str, List[Tuple[str, float]]]],
        page_shapes: List[Tuple[int, int]],
        crop_orientations: List[Dict[str, List[Dict[str, Any]]]],
        orientations: Optional[List[Dict[str, Any]]] = None,
        languages: Optional[List[Dict[str, Any]]] = None,
    ) -> KIEDocument:
        """Re-arrange detected words into structured predictions

        Args:
        ----
            pages: list of N elements, where each element represents the page image
            boxes: list of N dictionaries, where each element represents the localization predictions for a class,
                of shape (*, 5) or (*, 6) for all predictions
            objectness_scores: list of N dictionaries, where each element represents the objectness scores for a class
            text_preds: list of N dictionaries, where each element is the list of all word prediction
            page_shapes: shape of each page, of size N
            crop_orientations: list of N dictonaries, where each element is
                a list containing the general crop orientations (orientations + confidences) of the crops
            orientations: optional, list of N elements,
                where each element is a dictionary containing the orientation (orientation + confidence)
            languages: optional, list of N elements,
                where each element is a dictionary containing the language (language + confidence)

        Returns:
        -------
            document object
        """
        if len(boxes) != len(text_preds) != len(crop_orientations) != len(objectness_scores) or len(boxes) != len(
            page_shapes
        ) != len(crop_orientations) != len(objectness_scores):
            raise ValueError("All arguments are expected to be lists of the same size")
        _orientations = (
            orientations if isinstance(orientations, list) else [None] * len(boxes)  # type: ignore[list-item]
        )
        _languages = languages if isinstance(languages, list) else [None] * len(boxes)  # type: ignore[list-item]
        if self.export_as_straight_boxes and len(boxes) > 0:
            # If boxes are already straight OK, else fit a bounding rect
            if next(iter(boxes[0].values())).ndim == 3:
                straight_boxes: List[Dict[str, np.ndarray]] = []
                # Iterate over pages
                for p_boxes in boxes:
                    # Iterate over boxes of the pages
                    straight_boxes_dict = {}
                    for k, box in p_boxes.items():
                        straight_boxes_dict[k] = np.concatenate((box.min(1), box.max(1)), 1)
                    straight_boxes.append(straight_boxes_dict)
                boxes = straight_boxes

        _pages = [
            KIEPage(
                page,
                {
                    k: self._build_blocks(
                        page_boxes[k],
                        loc_scores[k],
                        word_preds[k],
                        word_crop_orientations[k],
                    )
                    for k in page_boxes.keys()
                },
                _idx,
                shape,
                orientation,
                language,
            )
            for page, _idx, shape, page_boxes, loc_scores, word_preds, word_crop_orientations, orientation, language in zip(  # noqa: E501
                pages,
                range(len(boxes)),
                page_shapes,
                boxes,
                objectness_scores,
                text_preds,
                crop_orientations,
                _orientations,
                _languages,
            )
        ]

        return KIEDocument(_pages)

    def _build_blocks(  # type: ignore[override]
        self,
        boxes: np.ndarray,
        objectness_scores: np.ndarray,
        word_preds: List[Tuple[str, float]],
        crop_orientations: List[Dict[str, Any]],
    ) -> List[Prediction]:
        """Gather independent words in structured blocks

        Args:
        ----
            boxes: bounding boxes of all detected words of the page, of shape (N, 4) or (N, 4, 2)
            objectness_scores: objectness scores of all detected words of the page
            word_preds: list of all detected words of the page, of shape N
            crop_orientations: list of orientations for each word crop

        Returns:
        -------
            list of block elements
        """
        if boxes.shape[0] != len(word_preds):
            raise ValueError(f"Incompatible argument lengths: {boxes.shape[0]}, {len(word_preds)}")

        if boxes.shape[0] == 0:
            return []

        # Decide whether we try to form lines
        _boxes = boxes
        idxs, _ = self._sort_boxes(_boxes if _boxes.ndim == 3 else _boxes[:, :4])
        predictions = [
            Prediction(
                value=word_preds[idx][0],
                confidence=word_preds[idx][1],
                geometry=tuple([tuple(pt) for pt in boxes[idx].tolist()]),  # type: ignore[arg-type]
                objectness_score=float(objectness_scores[idx]),
                crop_orientation=crop_orientations[idx],
            )
            if boxes.ndim == 3
            else Prediction(
                value=word_preds[idx][0],
                confidence=word_preds[idx][1],
                geometry=((boxes[idx, 0], boxes[idx, 1]), (boxes[idx, 2], boxes[idx, 3])),
                objectness_score=float(objectness_scores[idx]),
                crop_orientation=crop_orientations[idx],
            )
            for idx in idxs
        ]
        return predictions