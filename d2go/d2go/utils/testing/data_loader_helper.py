class LocalImageGenerator:
    def __init__(self, image_dir, width, height):
        self._width = width
        self._height = height
        self._image_dir = image_dir

    def get_image_dir(self):
        return self._image_dir

    def get_image_dict(self, i):
        return {
            "file_name": "{}.jpg".format(i),
            "width": self._width,
            "height": self._height,
            "id": i,
        }

    def prepare_image(self, i):
        image = Image.new("RGB", (self._width, self._height))
        image.save(os.path.join(self._image_dir, self.get_image_dict(i)["file_name"]))

