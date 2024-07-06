def slice_coco(
    coco_annotation_file_path: str,
    image_dir: str,
    output_coco_annotation_file_name: str,
    output_dir: Optional[str] = None,
    ignore_negative_samples: bool = False,
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    min_area_ratio: float = 0.1,
    out_ext: Optional[str] = None,
    verbose: bool = False,
) -> List[Union[Dict, str]]:
    """
    Slice large images given in a directory, into smaller windows. If out_name is given export sliced images and coco file.

    Args:
        coco_annotation_file_pat (str): Location of the coco annotation file
        image_dir (str): Base directory for the images
        output_coco_annotation_file_name (str): File name of the exported coco
            datatset json.
        output_dir (str, optional): Output directory
        ignore_negative_samples (bool): If True, images without annotations
            are ignored. Defaults to False.
        slice_height (int): Height of each slice. Default 512.
        slice_width (int): Width of each slice. Default 512.
        overlap_height_ratio (float): Fractional overlap in height of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        overlap_width_ratio (float): Fractional overlap in width of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        min_area_ratio (float): If the cropped annotation area to original annotation
            ratio is smaller than this value, the annotation is filtered out. Default 0.1.
        out_ext (str, optional): Extension of saved images. Default is the
            original suffix.
        verbose (bool, optional): Switch to print relevant values to screen.
            Default 'False'.

    Returns:
        coco_dict: dict
            COCO dict for sliced images and annotations
        save_path: str
            Path to the saved coco file
    """

    # read coco file
    coco_dict: Dict = load_json(coco_annotation_file_path)
    # create image_id_to_annotation_list mapping
    coco = Coco.from_coco_dict_or_path(coco_dict)
    # init sliced coco_utils.CocoImage list
    sliced_coco_images: List = []

    # iterate over images and slice
    for idx, coco_image in enumerate(tqdm(coco.images)):
        # get image path
        image_path: str = os.path.join(image_dir, coco_image.file_name)
        # get annotation json list corresponding to selected coco image
        # slice image
        try:
            slice_image_result = slice_image(
                image=image_path,
                coco_annotation_list=coco_image.annotations,
                output_file_name=f"{Path(coco_image.file_name).stem}_{idx}",
                output_dir=output_dir,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
                min_area_ratio=min_area_ratio,
                out_ext=out_ext,
                verbose=verbose,
            )
            # append slice outputs
            sliced_coco_images.extend(slice_image_result.coco_images)
        except TopologicalError:
            logger.warning(f"Invalid annotation found, skipping this image: {image_path}")

    # create and save coco dict
    coco_dict = create_coco_dict(
        sliced_coco_images, coco_dict["categories"], ignore_negative_samples=ignore_negative_samples
    )
    save_path = ""
    if output_coco_annotation_file_name and output_dir:
        save_path = Path(output_dir) / (output_coco_annotation_file_name + "_coco.json")
        save_json(coco_dict, save_path)

    return coco_dict, save_path

