def get_dataloader(data_root: str, mode: str = "val", batch_size: int = 4):
    # Prepare the datasets for training
    # Acquire the images and labels from the coco128 dataset
    dataset = get_dataset(data_root=data_root, mode=mode)

    # We adopt the sequential sampler in order to repeat the experiment
    sampler = torch.utils.data.SequentialSampler(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size,
        sampler=sampler,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    return loader

def get_dataset(data_root: str, mode: str = "val"):
    # Acquire the images and labels from the coco128 dataset
    data_path = Path(data_root)
    coco128_dirname = "coco128"
    coco128_path = data_path / coco128_dirname
    image_root = coco128_path / "images" / "train2017"
    annotation_file = coco128_path / "annotations" / "instances_train2017.json"

    if not annotation_file.is_file():
        prepare_coco128(data_path, dirname=coco128_dirname)

    if mode == "train":
        dataset = COCODetection(image_root, annotation_file, default_train_transforms())
    elif mode == "val":
        dataset = COCODetection(image_root, annotation_file, default_val_transforms())
    else:
        raise NotImplementedError(f"Currently not supports mode {mode}")

    return dataset

