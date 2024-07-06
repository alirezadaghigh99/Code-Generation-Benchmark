def create_test_photos(number_of_photos=1, **kwargs):
    return [create_test_photo(**kwargs) for _ in range(0, number_of_photos)]

def create_test_photo(**kwargs):
    pk = fake.md5()
    if "aspect_ratio" not in kwargs.keys():
        kwargs["aspect_ratio"] = 1
    photo = Photo(pk=pk, image_hash=pk, **kwargs)
    file = create_test_file(f"/tmp/{pk}.png", photo.owner, ONE_PIXEL_PNG)
    photo.main_file = file
    if "added_on" not in kwargs.keys():
        photo.added_on = timezone.now()
    photo.save()
    return photo

