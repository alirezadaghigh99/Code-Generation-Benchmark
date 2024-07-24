class BlockSampler(BaseSampler):
    """
    Sampler that returns 3D blocks from 3D images.
    """
    def __init__(self, block_size, stride, batch_size, shuffle=False):
        
        if isinstance(block_size, int):
            block_size = [block_size, block_size, block_size]
        
        if isinstance(stride, int):
            stride = [stride, stride, stride]
            
        self.block_size = block_size
        self.stride = stride
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __call__(self, x, y):
        # create patches of all images
        self.x, self.y = create_blocks(x, y, self.block_size, self.stride)
        
        xx = self.x[0]
        if isinstance(xx, list):
            while isinstance(xx, list):
                batch_length = len(xx)
                xx = xx[0]
        else:
            batch_length = len(self.x)
        
        self.n_batches = math.ceil(len(self.x) / self.batch_size)
        self.batch_length = batch_length
        
        return self

