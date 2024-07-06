def write_ply(
        filename: str,
        count: int,
        map_to_tensors: typing.OrderedDict[str, np.ndarray],
    ):
        """
        Writes a PLY file with given vertex properties and a tensor of float or uint8 values in the order specified by the OrderedDict.
        Note: All float values will be converted to float32 for writing.

        Parameters:
        filename (str): The name of the file to write.
        count (int): The number of vertices to write.
        map_to_tensors (OrderedDict[str, np.ndarray]): An ordered dictionary mapping property names to numpy arrays of float or uint8 values.
            Each array should be 1-dimensional and of equal length matching 'count'. Arrays should not be empty.
        """

        # Ensure count matches the length of all tensors
        if not all(len(tensor) == count for tensor in map_to_tensors.values()):
            raise ValueError("Count does not match the length of all tensors")

        # Type check for numpy arrays of type float or uint8 and non-empty
        if not all(
            isinstance(tensor, np.ndarray)
            and (tensor.dtype.kind == "f" or tensor.dtype == np.uint8)
            and tensor.size > 0
            for tensor in map_to_tensors.values()
        ):
            raise ValueError("All tensors must be numpy arrays of float or uint8 type and not empty")

        with open(filename, "wb") as ply_file:
            # Write PLY header
            ply_file.write(b"ply\n")
            ply_file.write(b"format binary_little_endian 1.0\n")

            ply_file.write(f"element vertex {count}\n".encode())

            # Write properties, in order due to OrderedDict
            for key, tensor in map_to_tensors.items():
                data_type = "float" if tensor.dtype.kind == "f" else "uchar"
                ply_file.write(f"property {data_type} {key}\n".encode())

            ply_file.write(b"end_header\n")

            # Write binary data
            # Note: If this is a performance bottleneck consider using numpy.hstack for efficiency improvement
            for i in range(count):
                for tensor in map_to_tensors.values():
                    value = tensor[i]
                    if tensor.dtype.kind == "f":
                        ply_file.write(np.float32(value).tobytes())
                    elif tensor.dtype == np.uint8:
                        ply_file.write(value.tobytes())

