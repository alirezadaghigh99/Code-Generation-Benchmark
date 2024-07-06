def create_slices(x, axis):
    def flatten_extend(matrix):
        flat_list = []
        for row in matrix:
            flat_list.extend(row)
        return flat_list
    
    if isinstance(x[0], list):
        return [create_slices([x[i][j] for i in range(len(x))], axis) for j in range(len(x[0]))]
    if ants.is_image(x[0]):
        return flatten_extend([[xx.slice_image(axis, i) for i in range(xx.shape[axis])] for xx in x])
    else:
        return x

