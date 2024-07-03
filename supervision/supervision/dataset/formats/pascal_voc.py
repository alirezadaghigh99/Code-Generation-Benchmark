def detections_from_xml_obj(
    root: Element, classes: List[str], resolution_wh, force_masks: bool = False
) -> Tuple[Detections, List[str]]:
    """
    Converts an XML object in Pascal VOC format to a Detections object.
    Expected XML format:
    <annotation>
        ...
        <object>
            <name>dog</name>
            <bndbox>
                <xmin>48</xmin>
                <ymin>240</ymin>
                <xmax>195</xmax>
                <ymax>371</ymax>
            </bndbox>
            <polygon>
                <x1>48</x1>
                <y1>240</y1>
                <x2>195</x2>
                <y2>240</y2>
                <x3>195</x3>
                <y3>371</y3>
                <x4>48</x4>
                <y4>371</y4>
            </polygon>
        </object>
    </annotation>

    Returns:
        Tuple[Detections, List[str]]: A tuple containing a Detections object and an
            updated list of class names, extended with the class names
            from the XML object.
    """
    xyxy = []
    class_names = []
    masks = []
    with_masks = False
    extended_classes = classes[:]
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        class_names.append(class_name)

        bbox = obj.find("bndbox")
        x1 = int(bbox.find("xmin").text)
        y1 = int(bbox.find("ymin").text)
        x2 = int(bbox.find("xmax").text)
        y2 = int(bbox.find("ymax").text)

        xyxy.append([x1, y1, x2, y2])

        with_masks = obj.find("polygon") is not None
        with_masks = force_masks if force_masks else with_masks

        for polygon in obj.findall("polygon"):
            polygon = parse_polygon_points(polygon)
            # https://github.com/roboflow/supervision/issues/144
            polygon -= 1

            mask_from_polygon = polygon_to_mask(
                polygon=polygon,
                resolution_wh=resolution_wh,
            )
            masks.append(mask_from_polygon)

    xyxy = np.array(xyxy) if len(xyxy) > 0 else np.empty((0, 4))

    # https://github.com/roboflow/supervision/issues/144
    xyxy -= 1

    for k in set(class_names):
        if k not in extended_classes:
            extended_classes.append(k)
    class_id = np.array(
        [extended_classes.index(class_name) for class_name in class_names]
    )

    annotation = Detections(
        xyxy=xyxy.astype(np.float32),
        mask=np.array(masks).astype(bool) if with_masks else None,
        class_id=class_id,
    )

    return annotation, extended_classes