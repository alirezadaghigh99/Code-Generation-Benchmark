def from_coco_dict_or_path(
        cls,
        coco_dict_or_path: Union[Dict, str],
        image_dir: Optional[str] = None,
        remapping_dict: Optional[Dict] = None,
        ignore_negative_samples: bool = False,
        clip_bboxes_to_img_dims: bool = False,
        use_threads: bool = False,
        num_threads: int = 10,
    ):
        """
        Creates coco object from COCO formatted dict or COCO dataset file path.

        Args:
            coco_dict_or_path: dict/str or List[dict/str]
                COCO formatted dict or COCO dataset file path
                List of COCO formatted dict or COCO dataset file path
            image_dir: str
                Base file directory that contains dataset images. Required for merging and yolov5 conversion.
            remapping_dict: dict
                {1:0, 2:1} maps category id 1 to 0 and category id 2 to 1
            ignore_negative_samples: bool
                If True ignores images without annotations in all operations.
            clip_bboxes_to_img_dims: bool = False
                Limits bounding boxes to image dimensions.
            use_threads: bool = False
                Use threads when processing the json image list, defaults to False
            num_threads: int = 10
                Slice the image list to given number of chunks, defaults to 10

        Properties:
            images: list of CocoImage
            category_mapping: dict
        """
        # init coco object
        coco = cls(
            image_dir=image_dir,
            remapping_dict=remapping_dict,
            ignore_negative_samples=ignore_negative_samples,
            clip_bboxes_to_img_dims=clip_bboxes_to_img_dims,
        )

        if type(coco_dict_or_path) not in [str, dict]:
            raise TypeError("coco_dict_or_path should be a dict or str")

        # load coco dict if path is given
        if type(coco_dict_or_path) == str:
            coco_dict = load_json(coco_dict_or_path)
        else:
            coco_dict = coco_dict_or_path

        dict_size = len(coco_dict["images"])

        # arrange image id to annotation id mapping
        coco.add_categories_from_coco_category_list(coco_dict["categories"])
        image_id_to_annotation_list = get_imageid2annotationlist_mapping(coco_dict)
        category_mapping = coco.category_mapping

        # https://github.com/obss/sahi/issues/98
        image_id_set: Set = set()

        lock = Lock()

        def fill_image_id_set(start, finish, image_list, _image_id_set, _image_id_to_annotation_list, _coco, lock):
            for coco_image_dict in tqdm(
                image_list[start:finish], f"Loading coco annotations between {start} and {finish}"
            ):
                coco_image = CocoImage.from_coco_image_dict(coco_image_dict)
                image_id = coco_image_dict["id"]
                # https://github.com/obss/sahi/issues/98
                if image_id in _image_id_set:
                    print(f"duplicate image_id: {image_id}, will be ignored.")
                    continue
                else:
                    lock.acquire()
                    _image_id_set.add(image_id)
                    lock.release()

                # select annotations of the image
                annotation_list = _image_id_to_annotation_list[image_id]
                for coco_annotation_dict in annotation_list:
                    # apply category remapping if remapping_dict is provided
                    if _coco.remapping_dict is not None:
                        # apply category remapping (id:id)
                        category_id = _coco.remapping_dict[coco_annotation_dict["category_id"]]
                        # update category id
                        coco_annotation_dict["category_id"] = category_id
                    else:
                        category_id = coco_annotation_dict["category_id"]
                    # get category name (id:name)
                    category_name = category_mapping[category_id]
                    coco_annotation = CocoAnnotation.from_coco_annotation_dict(
                        category_name=category_name, annotation_dict=coco_annotation_dict
                    )
                    coco_image.add_annotation(coco_annotation)
                _coco.add_image(coco_image)

        chunk_size = dict_size / num_threads

        if use_threads is True:
            for i in range(num_threads):
                start = i * chunk_size
                finish = start + chunk_size
                if finish > dict_size:
                    finish = dict_size
                t = Thread(
                    target=fill_image_id_set,
                    args=(start, finish, coco_dict["images"], image_id_set, image_id_to_annotation_list, coco, lock),
                )
                t.start()

            main_thread = threading.currentThread()
            for t in threading.enumerate():
                if t is not main_thread:
                    t.join()

        else:
            for coco_image_dict in tqdm(coco_dict["images"], "Loading coco annotations"):
                coco_image = CocoImage.from_coco_image_dict(coco_image_dict)
                image_id = coco_image_dict["id"]
                # https://github.com/obss/sahi/issues/98
                if image_id in image_id_set:
                    print(f"duplicate image_id: {image_id}, will be ignored.")
                    continue
                else:
                    image_id_set.add(image_id)
                # select annotations of the image
                annotation_list = image_id_to_annotation_list[image_id]
                for coco_annotation_dict in annotation_list:
                    # apply category remapping if remapping_dict is provided
                    if coco.remapping_dict is not None:
                        # apply category remapping (id:id)
                        category_id = coco.remapping_dict[coco_annotation_dict["category_id"]]
                        # update category id
                        coco_annotation_dict["category_id"] = category_id
                    else:
                        category_id = coco_annotation_dict["category_id"]
                    # get category name (id:name)
                    category_name = category_mapping[category_id]
                    coco_annotation = CocoAnnotation.from_coco_annotation_dict(
                        category_name=category_name, annotation_dict=coco_annotation_dict
                    )
                    coco_image.add_annotation(coco_annotation)
                coco.add_image(coco_image)

        if clip_bboxes_to_img_dims:
            coco = coco.get_coco_with_clipped_bboxes()
        return coco

