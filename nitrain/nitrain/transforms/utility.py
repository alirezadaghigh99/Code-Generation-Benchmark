class NumpyFunction(BaseTransform):
    """
    Apply a user-supplied function operating on a numpy array
    """
    
    def __init__(self, fn, **kwargs):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.NumpyFunction(np.clip, a_min=10, a_max=100)
        img2 = mytx(img)
        """
        self.fn = fn
        self.kwargs = kwargs
        
    def __call__(self, *images):
        images = [ants.from_numpy_like(self.fn(image.numpy(), **self.kwargs), image) for image in images]
        return images if len(images) > 1 else images[0]

