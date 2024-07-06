def save_ply(
    f,
    verts: torch.Tensor,
    faces: Optional[torch.LongTensor] = None,
    verts_normals: Optional[torch.Tensor] = None,
    ascii: bool = False,
    decimal_places: Optional[int] = None,
    path_manager: Optional[PathManager] = None,
) -> None:
    """
    Save a mesh to a .ply file.

    Args:
        f: File (or path) to which the mesh should be written.
        verts: FloatTensor of shape (V, 3) giving vertex coordinates.
        faces: LongTensor of shape (F, 3) giving faces.
        verts_normals: FloatTensor of shape (V, 3) giving vertex normals.
        ascii: (bool) whether to use the ascii ply format.
        decimal_places: Number of decimal places for saving if ascii=True.
        path_manager: PathManager for interpreting f if it is a str.
    """

    if len(verts) and not (verts.dim() == 2 and verts.size(1) == 3):
        message = "Argument 'verts' should either be empty or of shape (num_verts, 3)."
        raise ValueError(message)

    if (
        faces is not None
        and len(faces)
        and not (faces.dim() == 2 and faces.size(1) == 3)
    ):
        message = "Argument 'faces' should either be empty or of shape (num_faces, 3)."
        raise ValueError(message)

    if (
        verts_normals is not None
        and len(verts_normals)
        and not (
            verts_normals.dim() == 2
            and verts_normals.size(1) == 3
            and verts_normals.size(0) == verts.size(0)
        )
    ):
        message = "Argument 'verts_normals' should either be empty or of shape (num_verts, 3)."
        raise ValueError(message)

    if path_manager is None:
        path_manager = PathManager()
    with _open_file(f, path_manager, "wb") as f:
        _save_ply(
            f,
            verts=verts,
            faces=faces,
            verts_normals=verts_normals,
            verts_colors=None,
            ascii=ascii,
            decimal_places=decimal_places,
            colors_as_uint8=False,
        )

def load_ply(
    f, *, path_manager: Optional[PathManager] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load the verts and faces from a .ply file.
    Note that the preferred way to load data from such a file
    is to use the IO.load_mesh and IO.load_pointcloud functions,
    which can read more of the data.

    Example .ply file format::

        ply
        format ascii 1.0           { ascii/binary, format version number }
        comment made by Greg Turk  { comments keyword specified, like all lines }
        comment this file is a cube
        element vertex 8           { define "vertex" element, 8 of them in file }
        property float x           { vertex contains float "x" coordinate }
        property float y           { y coordinate is also a vertex property }
        property float z           { z coordinate, too }
        element face 6             { there are 6 "face" elements in the file }
        property list uchar int vertex_index { "vertex_indices" is a list of ints }
        end_header                 { delimits the end of the header }
        0 0 0                      { start of vertex list }
        0 0 1
        0 1 1
        0 1 0
        1 0 0
        1 0 1
        1 1 1
        1 1 0
        4 0 1 2 3                  { start of face list }
        4 7 6 5 4
        4 0 4 5 1
        4 1 5 6 2
        4 2 6 7 3
        4 3 7 4 0

    Args:
        f:  A binary or text file-like object (with methods read, readline,
            tell and seek), a pathlib path or a string containing a file name.
            If the ply file is in the binary ply format rather than the text
            ply format, then a text stream is not supported.
            It is easiest to use a binary stream in all cases.
        path_manager: PathManager for loading if f is a str.

    Returns:
        verts: FloatTensor of shape (V, 3).
        faces: LongTensor of vertex indices, shape (F, 3).
    """

    if path_manager is None:
        path_manager = PathManager()
    data = _load_ply(f, path_manager=path_manager)
    faces = data.faces
    if faces is None:
        faces = torch.zeros(0, 3, dtype=torch.int64)

    return data.verts, faces

