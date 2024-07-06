def get_fast_benchmark(
    use_task_labels=False,
    shuffle=True,
    n_samples_per_class=100,
    n_classes=10,
    n_features=6,
    seed=None,
    train_transform=None,
    eval_transform=None,
):
    train, test = dummy_classification_datasets(
        n_classes, n_features, n_samples_per_class, seed
    )
    my_nc_benchmark = nc_benchmark(
        train,
        test,
        5,
        task_labels=use_task_labels,
        shuffle=shuffle,
        train_transform=train_transform,
        eval_transform=eval_transform,
        seed=seed,
    )
    return my_nc_benchmark

def dummy_image_dataset():
    """Returns a PyTorch image dataset of 10 classes."""
    global image_data

    if image_data is None:
        image_data = MNIST(
            root=default_dataset_location("mnist"),
            train=True,
            download=True,
        )
    return image_data

def get_fast_detection_datasets(
    n_images=30,
    max_elements_per_image=10,
    n_samples_per_class=20,
    n_classes=10,
    seed=None,
    image_size=64,
    n_test_images=5,
):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    assert n_images * max_elements_per_image >= n_samples_per_class * n_classes
    assert n_test_images < n_images
    assert n_test_images > 0

    base_n_per_images = (n_samples_per_class * n_classes) // n_images
    additional_elements = (n_samples_per_class * n_classes) % n_images
    to_allocate = np.full(n_images, base_n_per_images)
    to_allocate[:additional_elements] += 1
    np.random.shuffle(to_allocate)
    classes_elements = np.repeat(np.arange(n_classes), n_samples_per_class)
    np.random.shuffle(classes_elements)

    import matplotlib.colors as mcolors

    forms = ["ellipse", "rectangle", "line", "arc"]
    colors = list(mcolors.TABLEAU_COLORS.values())
    combs = list(itertools.product(forms, colors))
    random.shuffle(combs)

    generated_images = []
    generated_targets = []
    for img_idx in range(n_images):
        n_to_allocate = to_allocate[img_idx]
        base_alloc_idx = to_allocate[:img_idx].sum()
        classes_to_instantiate = classes_elements[
            base_alloc_idx : base_alloc_idx + n_to_allocate
        ]

        _, _, clusters = make_blobs(
            n_to_allocate,
            n_features=2,
            centers=n_to_allocate,
            center_box=(0, image_size - 1),
            random_state=seed,
            return_centers=True,
        )

        from PIL import Image as ImageApi
        from PIL import ImageDraw

        im = ImageApi.new("RGB", (image_size, image_size))
        draw = ImageDraw.Draw(im)

        target = {
            "boxes": torch.zeros((n_to_allocate, 4), dtype=torch.float32),
            "labels": torch.zeros((n_to_allocate,), dtype=torch.long),
            "image_id": torch.full((1,), img_idx, dtype=torch.long),
            "area": torch.zeros((n_to_allocate,), dtype=torch.float32),
            "iscrowd": torch.zeros((n_to_allocate,), dtype=torch.long),
        }

        obj_sizes = np.random.uniform(
            low=image_size * 0.1 * 0.95,
            high=image_size * 0.1 * 1.05,
            size=(n_to_allocate,),
        )
        for center_idx, center in enumerate(clusters):
            obj_size = float(obj_sizes[center_idx])
            class_to_gen = classes_to_instantiate[center_idx]

            class_form, class_color = combs[class_to_gen]

            left = center[0] - obj_size
            top = center[1] - obj_size
            right = center[0] + obj_size
            bottom = center[1] + obj_size
            ltrb = (left, top, right, bottom)
            if class_form == "ellipse":
                draw.ellipse(ltrb, fill=class_color)
            elif class_form == "rectangle":
                draw.rectangle(ltrb, fill=class_color)
            elif class_form == "line":
                draw.line(ltrb, fill=class_color, width=max(1, int(obj_size * 0.25)))
            elif class_form == "arc":
                draw.arc(ltrb, fill=class_color, start=45, end=200)
            else:
                raise RuntimeError("Unsupported form")

            target["boxes"][center_idx] = torch.as_tensor(ltrb)
            target["labels"][center_idx] = class_to_gen
            target["area"][center_idx] = obj_size**2

        generated_images.append(np.array(im))
        generated_targets.append(target)
        im.close()

    test_indices = set(
        np.random.choice(n_images, n_test_images, replace=False).tolist()
    )
    train_images = [x for i, x in enumerate(generated_images) if i not in test_indices]
    test_images = [x for i, x in enumerate(generated_images) if i in test_indices]

    train_targets = [
        x for i, x in enumerate(generated_targets) if i not in test_indices
    ]
    test_targets = [x for i, x in enumerate(generated_targets) if i in test_indices]

    return make_detection_dataset(
        _DummyDetectionDataset(train_images, train_targets),
        targets=train_targets,
        task_labels=0,
    ), make_detection_dataset(
        _DummyDetectionDataset(test_images, test_targets),
        targets=test_targets,
        task_labels=0,
    )

