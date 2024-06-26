class Objects365V1Dataset(CocoDataset):
    """Objects365 v1 dataset for detection."""

    METAINFO = {
        'classes':
        ('person', 'sneakers', 'chair', 'hat', 'lamp', 'bottle',
         'cabinet/shelf', 'cup', 'car', 'glasses', 'picture/frame', 'desk',
         'handbag', 'street lights', 'book', 'plate', 'helmet',
         'leather shoes', 'pillow', 'glove', 'potted plant', 'bracelet',
         'flower', 'tv', 'storage box', 'vase', 'bench', 'wine glass', 'boots',
         'bowl', 'dining table', 'umbrella', 'boat', 'flag', 'speaker',
         'trash bin/can', 'stool', 'backpack', 'couch', 'belt', 'carpet',
         'basket', 'towel/napkin', 'slippers', 'barrel/bucket', 'coffee table',
         'suv', 'toy', 'tie', 'bed', 'traffic light', 'pen/pencil',
         'microphone', 'sandals', 'canned', 'necklace', 'mirror', 'faucet',
         'bicycle', 'bread', 'high heels', 'ring', 'van', 'watch', 'sink',
         'horse', 'fish', 'apple', 'camera', 'candle', 'teddy bear', 'cake',
         'motorcycle', 'wild bird', 'laptop', 'knife', 'traffic sign',
         'cell phone', 'paddle', 'truck', 'cow', 'power outlet', 'clock',
         'drum', 'fork', 'bus', 'hanger', 'nightstand', 'pot/pan', 'sheep',
         'guitar', 'traffic cone', 'tea pot', 'keyboard', 'tripod', 'hockey',
         'fan', 'dog', 'spoon', 'blackboard/whiteboard', 'balloon',
         'air conditioner', 'cymbal', 'mouse', 'telephone', 'pickup truck',
         'orange', 'banana', 'airplane', 'luggage', 'skis', 'soccer',
         'trolley', 'oven', 'remote', 'baseball glove', 'paper towel',
         'refrigerator', 'train', 'tomato', 'machinery vehicle', 'tent',
         'shampoo/shower gel', 'head phone', 'lantern', 'donut',
         'cleaning products', 'sailboat', 'tangerine', 'pizza', 'kite',
         'computer box', 'elephant', 'toiletries', 'gas stove', 'broccoli',
         'toilet', 'stroller', 'shovel', 'baseball bat', 'microwave',
         'skateboard', 'surfboard', 'surveillance camera', 'gun', 'life saver',
         'cat', 'lemon', 'liquid soap', 'zebra', 'duck', 'sports car',
         'giraffe', 'pumpkin', 'piano', 'stop sign', 'radiator', 'converter',
         'tissue ', 'carrot', 'washing machine', 'vent', 'cookies',
         'cutting/chopping board', 'tennis racket', 'candy',
         'skating and skiing shoes', 'scissors', 'folder', 'baseball',
         'strawberry', 'bow tie', 'pigeon', 'pepper', 'coffee machine',
         'bathtub', 'snowboard', 'suitcase', 'grapes', 'ladder', 'pear',
         'american football', 'basketball', 'potato', 'paint brush', 'printer',
         'billiards', 'fire hydrant', 'goose', 'projector', 'sausage',
         'fire extinguisher', 'extension cord', 'facial mask', 'tennis ball',
         'chopsticks', 'electronic stove and gas stove', 'pie', 'frisbee',
         'kettle', 'hamburger', 'golf club', 'cucumber', 'clutch', 'blender',
         'tong', 'slide', 'hot dog', 'toothbrush', 'facial cleanser', 'mango',
         'deer', 'egg', 'violin', 'marker', 'ship', 'chicken', 'onion',
         'ice cream', 'tape', 'wheelchair', 'plum', 'bar soap', 'scale',
         'watermelon', 'cabbage', 'router/modem', 'golf ball', 'pine apple',
         'crane', 'fire truck', 'peach', 'cello', 'notepaper', 'tricycle',
         'toaster', 'helicopter', 'green beans', 'brush', 'carriage', 'cigar',
         'earphone', 'penguin', 'hurdle', 'swing', 'radio', 'CD',
         'parking meter', 'swan', 'garlic', 'french fries', 'horn', 'avocado',
         'saxophone', 'trumpet', 'sandwich', 'cue', 'kiwi fruit', 'bear',
         'fishing rod', 'cherry', 'tablet', 'green vegetables', 'nuts', 'corn',
         'key', 'screwdriver', 'globe', 'broom', 'pliers', 'volleyball',
         'hammer', 'eggplant', 'trophy', 'dates', 'board eraser', 'rice',
         'tape measure/ruler', 'dumbbell', 'hamimelon', 'stapler', 'camel',
         'lettuce', 'goldfish', 'meat balls', 'medal', 'toothpaste',
         'antelope', 'shrimp', 'rickshaw', 'trombone', 'pomegranate',
         'coconut', 'jellyfish', 'mushroom', 'calculator', 'treadmill',
         'butterfly', 'egg tart', 'cheese', 'pig', 'pomelo', 'race car',
         'rice cooker', 'tuba', 'crosswalk sign', 'papaya', 'hair drier',
         'green onion', 'chips', 'dolphin', 'sushi', 'urinal', 'donkey',
         'electric drill', 'spring rolls', 'tortoise/turtle', 'parrot',
         'flute', 'measuring cup', 'shark', 'steak', 'poker card',
         'binoculars', 'llama', 'radish', 'noodles', 'yak', 'mop', 'crab',
         'microscope', 'barbell', 'bread/bun', 'baozi', 'lion', 'red cabbage',
         'polar bear', 'lighter', 'seal', 'mangosteen', 'comb', 'eraser',
         'pitaya', 'scallop', 'pencil case', 'saw', 'table tennis paddle',
         'okra', 'starfish', 'eagle', 'monkey', 'durian', 'game board',
         'rabbit', 'french horn', 'ambulance', 'asparagus', 'hoverboard',
         'pasta', 'target', 'hotair balloon', 'chainsaw', 'lobster', 'iron',
         'flashlight'),
        'palette':
        None
    }

    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)

        # 'categories' list in objects365_train.json and objects365_val.json
        # is inconsistent, need sort list(or dict) before get cat_ids.
        cats = self.coco.cats
        sorted_cats = {i: cats[i] for i in sorted(cats)}
        self.coco.cats = sorted_cats
        categories = self.coco.dataset['categories']
        sorted_categories = sorted(categories, key=lambda i: i['id'])
        self.coco.dataset['categories'] = sorted_categories
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list