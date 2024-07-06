def save_array(
    save_dir: str,
    arr: Union[np.ndarray, tf.Tensor],
    name: str,
    normalize: bool,
    save_nifti: bool = True,
    save_png: bool = True,
    overwrite: bool = True,
):
    """
    :param save_dir: path of the directory to save
    :param arr: 3D or 4D array to be saved
    :param name: name of the array, e.g. image, label, etc.
    :param normalize: true if the array's value has to be normalized when saving pngs,
        false means the value is between [0, 1].
    :param save_nifti: if true, array will be saved in nifti
    :param save_png: if true, array will be saved in png
    :param overwrite: if false, will not save the file in case the file exists
    """
    if isinstance(arr, tf.Tensor):
        arr = arr.numpy()
    if len(arr.shape) not in [3, 4]:
        raise ValueError(f"arr must be 3d or 4d numpy array or tf tensor, got {arr}")
    is_4d = len(arr.shape) == 4
    if is_4d:
        # if 4D array, it must be 3 channels
        if arr.shape[3] != 3:
            raise ValueError(
                f"4d arr must have 3 channels as last dimension, "
                f"got arr.shape = {arr.shape}"
            )

    # save in nifti format
    if save_nifti:
        nifti_file_path = os.path.join(save_dir, name + ".nii.gz")
        if overwrite or (not os.path.exists(nifti_file_path)):
            # save only if need to overwrite or doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            # output with Nifti1Image can be loaded by
            # - https://www.slicer.org/
            # - http://www.itksnap.org/
            # - http://ric.uthscsa.edu/mango/
            # However, outputs with Nifti2Image couldn't be loaded
            nib.save(
                img=nib.Nifti1Image(arr, affine=np.eye(4)), filename=nifti_file_path
            )

    # save in png
    if save_png:
        png_dir = os.path.join(save_dir, name)
        dir_existed = os.path.exists(png_dir)
        if normalize:
            # normalize arr such that it has only values between 0, 1
            arr = normalize_array(arr=arr)
        for depth_index in range(arr.shape[2]):
            png_file_path = os.path.join(png_dir, f"depth{depth_index}_{name}.png")
            if overwrite or (not os.path.exists(png_file_path)):
                if not dir_existed:
                    os.makedirs(png_dir, exist_ok=True)
                plt.imsave(
                    fname=png_file_path,
                    arr=arr[:, :, depth_index, :] if is_4d else arr[:, :, depth_index],
                    vmin=0,
                    vmax=1,
                    cmap="PiYG" if is_4d else "gray",
                )

