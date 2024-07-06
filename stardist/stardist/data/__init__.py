def test_image_nuclei_2d(return_mask=False):
    """ Fluorescence microscopy image and mask from the 2018 kaggle DSB challenge

    Caicedo et al. "Nucleus segmentation across imaging experiments: the 2018 Data Science Bowl." Nature methods 16.12
    """
    from tifffile import imread
    img = imread(abspath("images/img2d.tif"))
    mask = imread(abspath("images/mask2d.tif"))
    if return_mask:
        return img, mask
    else:
        return img

