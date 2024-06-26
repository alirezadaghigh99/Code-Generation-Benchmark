def draw_boxes(boxes: np.ndarray, image: np.ndarray, color: Optional[Tuple[int, int, int]] = None, **kwargs) -> None:
    """Draw an array of relative straight boxes on an image

    Args:
    ----
        boxes: array of relative boxes, of shape (*, 4)
        image: np array, float32 or uint8
        color: color to use for bounding box edges
        **kwargs: keyword arguments from `matplotlib.pyplot.plot`
    """
    h, w = image.shape[:2]
    # Convert boxes to absolute coords
    _boxes = deepcopy(boxes)
    _boxes[:, [0, 2]] *= w
    _boxes[:, [1, 3]] *= h
    _boxes = _boxes.astype(np.int32)
    for box in _boxes.tolist():
        xmin, ymin, xmax, ymax = box
        image = cv2.rectangle(
            image, (xmin, ymin), (xmax, ymax), color=color if isinstance(color, tuple) else (0, 0, 255), thickness=2
        )
    plt.imshow(image)
    plt.plot(**kwargs)