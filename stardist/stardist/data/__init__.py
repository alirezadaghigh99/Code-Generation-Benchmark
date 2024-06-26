def test_image_he_2d():
    """ H&E stained RGB example image from the Cancer Imaging Archive
    https://www.cancerimagingarchive.net
    """
    from imageio import imread
    img = imread(abspath("images/histo.jpg"))
    return img