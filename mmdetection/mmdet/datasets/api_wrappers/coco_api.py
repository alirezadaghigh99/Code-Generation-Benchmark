class COCOPanoptic(COCO):
    """This wrapper is for loading the panoptic style annotation file.

    The format is shown in the CocoPanopticDataset class.

    Args:
        annotation_file (str, optional): Path of annotation file.
            Defaults to None.
    """

    def __init__(self, annotation_file: Optional[str] = None) -> None:
        super(COCOPanoptic, self).__init__(annotation_file)

    def createIndex(self) -> None:
        """Create index."""
        # create index
        print('creating index...')
        # anns stores 'segment_id -> annotation'
        anns, cats, imgs = {}, {}, {}
        img_to_anns, cat_to_imgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                for seg_ann in ann['segments_info']:
                    # to match with instance.json
                    seg_ann['image_id'] = ann['image_id']
                    img_to_anns[ann['image_id']].append(seg_ann)
                    # segment_id is not unique in coco dataset orz...
                    # annotations from different images but
                    # may have same segment_id
                    if seg_ann['id'] in anns.keys():
                        anns[seg_ann['id']].append(seg_ann)
                    else:
                        anns[seg_ann['id']] = [seg_ann]

            # filter out annotations from other images
            img_to_anns_ = defaultdict(list)
            for k, v in img_to_anns.items():
                img_to_anns_[k] = [x for x in v if x['image_id'] == k]
            img_to_anns = img_to_anns_

        if 'images' in self.dataset:
            for img_info in self.dataset['images']:
                img_info['segm_file'] = img_info['file_name'].replace(
                    '.jpg', '.png')
                imgs[img_info['id']] = img_info

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                for seg_ann in ann['segments_info']:
                    cat_to_imgs[seg_ann['category_id']].append(ann['image_id'])

        print('index created!')

        self.anns = anns
        self.imgToAnns = img_to_anns
        self.catToImgs = cat_to_imgs
        self.imgs = imgs
        self.cats = cats

    def load_anns(self,
                  ids: Union[List[int], int] = []) -> Optional[List[dict]]:
        """Load anns with the specified ids.

        ``self.anns`` is a list of annotation lists instead of a
        list of annotations.

        Args:
            ids (Union[List[int], int]): Integer ids specifying anns.

        Returns:
            anns (List[dict], optional): Loaded ann objects.
        """
        anns = []

        if hasattr(ids, '__iter__') and hasattr(ids, '__len__'):
            # self.anns is a list of annotation lists instead of
            # a list of annotations
            for id in ids:
                anns += self.anns[id]
            return anns
        elif type(ids) == int:
            return self.anns[ids]

