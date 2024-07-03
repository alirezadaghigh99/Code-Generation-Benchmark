def visualize_page(
    page: Dict[str, Any],
    image: np.ndarray,
    words_only: bool = True,
    display_artefacts: bool = True,
    scale: float = 10,
    interactive: bool = True,
    add_labels: bool = True,
    **kwargs: Any,
) -> Figure:
    """Visualize a full page with predicted blocks, lines and words

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from doctr.utils.visualization import visualize_page
    >>> from doctr.models import ocr_db_crnn
    >>> model = ocr_db_crnn(pretrained=True)
    >>> input_page = (255 * np.random.rand(600, 800, 3)).astype(np.uint8)
    >>> out = model([[input_page]])
    >>> visualize_page(out[0].pages[0].export(), input_page)
    >>> plt.show()

    Args:
    ----
        page: the exported Page of a Document
        image: np array of the page, needs to have the same shape than page['dimensions']
        words_only: whether only words should be displayed
        display_artefacts: whether artefacts should be displayed
        scale: figsize of the largest windows side
        interactive: whether the plot should be interactive
        add_labels: for static plot, adds text labels on top of bounding box
        **kwargs: keyword arguments for the polygon patch

    Returns:
    -------
        the matplotlib figure
    """
    # Get proper scale and aspect ratio
    h, w = image.shape[:2]
    size = (scale * w / h, scale) if h > w else (scale, h / w * scale)
    fig, ax = plt.subplots(figsize=size)
    # Display the image
    ax.imshow(image)
    # hide both axis
    ax.axis("off")

    if interactive:
        artists: List[patches.Patch] = []  # instantiate an empty list of patches (to be drawn on the page)

    for block in page["blocks"]:
        if not words_only:
            rect = create_obj_patch(
                block["geometry"], page["dimensions"], label="block", color=(0, 1, 0), linewidth=1, **kwargs
            )
            # add patch on figure
            ax.add_patch(rect)
            if interactive:
                # add patch to cursor's artists
                artists.append(rect)

        for line in block["lines"]:
            if not words_only:
                rect = create_obj_patch(
                    line["geometry"], page["dimensions"], label="line", color=(1, 0, 0), linewidth=1, **kwargs
                )
                ax.add_patch(rect)
                if interactive:
                    artists.append(rect)

            for word in line["words"]:
                rect = create_obj_patch(
                    word["geometry"],
                    page["dimensions"],
                    label=f"{word['value']} (confidence: {word['confidence']:.2%})",
                    color=(0, 0, 1),
                    **kwargs,
                )
                ax.add_patch(rect)
                if interactive:
                    artists.append(rect)
                elif add_labels:
                    if len(word["geometry"]) == 5:
                        text_loc = (
                            int(page["dimensions"][1] * (word["geometry"][0] - word["geometry"][2] / 2)),
                            int(page["dimensions"][0] * (word["geometry"][1] - word["geometry"][3] / 2)),
                        )
                    else:
                        text_loc = (
                            int(page["dimensions"][1] * word["geometry"][0][0]),
                            int(page["dimensions"][0] * word["geometry"][0][1]),
                        )

                    if len(word["geometry"]) == 2:
                        # We draw only if boxes are in straight format
                        ax.text(
                            *text_loc,
                            word["value"],
                            size=10,
                            alpha=0.5,
                            color=(0, 0, 1),
                        )

        if display_artefacts:
            for artefact in block["artefacts"]:
                rect = create_obj_patch(
                    artefact["geometry"],
                    page["dimensions"],
                    label="artefact",
                    color=(0.5, 0.5, 0.5),
                    linewidth=1,
                    **kwargs,
                )
                ax.add_patch(rect)
                if interactive:
                    artists.append(rect)

    if interactive:
        import mplcursors

        # Create mlp Cursor to hover patches in artists
        mplcursors.Cursor(artists, hover=2).connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))
    fig.tight_layout(pad=0.0)

    return fig