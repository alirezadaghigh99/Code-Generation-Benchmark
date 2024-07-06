def infer_reader(x):
    """
    Infer reader from user-supplied values
    """
    if isinstance(x, list):
        if ants.is_image(x[0]):
            return readers.MemoryReader(x)
        elif np.isscalar(x[0]):
            return readers.MemoryReader(x)
        else:
            return readers.ComposeReader([infer_reader(xx) for xx in x])
            
    elif isinstance(x, dict):
        new_readers = []
        for key, value in x.items():
            value = infer_reader(value)
            value.label = key
            new_readers.append(value)
        if len(new_readers) > 1:
            return readers.ComposeReader(new_readers)
        else:
            return new_readers[0]
        
    elif isinstance(x, np.ndarray):
        return readers.MemoryReader(x)
    
    elif is_reader(x):
        return x
    
    raise Exception(f'Could not infer a configuration from given value: {x}')

