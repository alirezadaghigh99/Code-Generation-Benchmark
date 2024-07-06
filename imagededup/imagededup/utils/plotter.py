def _validate_args(
    image_dir: Union[PurePath, str], duplicate_map: Dict, filename: str
) -> PurePath:
    """Argument validator for plot_duplicates() defined below.
    Return PurePath to the image directory"""

    image_dir = Path(image_dir)
    assert (
        image_dir.is_dir()
    ), 'Provided image directory does not exist! Please provide the image directory where all files are present!'

    if not isinstance(duplicate_map, dict):
        raise ValueError('Please provide a valid Duplicate map!')
    if filename not in duplicate_map.keys():
        raise ValueError(
            'Please provide a valid filename present as a key in the duplicate_map!'
        )
    return image_dir

def plot_duplicates(
    image_dir: Union[PurePath, str],
    duplicate_map: Dict,
    filename: str,
    outfile: str = None,
) -> None:
    """
    Given filename for an image, plot duplicates along with the original image using the duplicate map obtained using
    find_duplicates method.

    Args:
        image_dir: image directory where all files in duplicate_map are present.
        duplicate_map: mapping of filename to found duplicates (could be with or without scores).
        filename: Name of the file for which duplicates are to be plotted, must be a key in the duplicate_map.
        dictionary.
        outfile: Optional, name of the file to save the plot. Default is None.

    Example:
    ```
        from imagededup.utils import plot_duplicates
        plot_duplicates(image_dir='path/to/image/directory',
                        duplicate_map=duplicate_map,
                        filename='path/to/image.jpg')
    ```
    """
    # validate args
    image_dir = _validate_args(image_dir=image_dir, duplicate_map=duplicate_map, filename=filename)

    retrieved = duplicate_map[filename]
    assert len(retrieved) != 0, 'Provided filename has no duplicates!'

    # plot
    if isinstance(retrieved[0], tuple):
        _plot_images(
            image_dir=image_dir,
            orig=filename,
            image_list=retrieved,
            scores=True,
            outfile=outfile,
        )
    else:
        _plot_images(
            image_dir=image_dir,
            orig=filename,
            image_list=retrieved,
            scores=False,
            outfile=outfile,
        )

def _formatter(val: Union[int, np.float32]):
    """
    For printing floats only upto 3rd precision. Ints are unchanged.
    """
    if isinstance(val, np.float32):
        return f'{val:.3f}'
    else:
        return val

