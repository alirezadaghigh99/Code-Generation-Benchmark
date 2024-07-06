def load_obj(
    f,
    load_textures: bool = True,
    create_texture_atlas: bool = False,
    texture_atlas_size: int = 4,
    texture_wrap: Optional[str] = "repeat",
    device: Device = "cpu",
    path_manager: Optional[PathManager] = None,
):
    """
    Load a mesh from a .obj file and optionally textures from a .mtl file.
    Currently this handles verts, faces, vertex texture uv coordinates, normals,
    texture images and material reflectivity values.

    Note .obj files are 1-indexed. The tensors returned from this function
    are 0-indexed. OBJ spec reference: http://www.martinreddy.net/gfx/3d/OBJ.spec

    Example .obj file format:
    ::
        # this is a comment
        v 1.000000 -1.000000 -1.000000
        v 1.000000 -1.000000 1.000000
        v -1.000000 -1.000000 1.000000
        v -1.000000 -1.000000 -1.000000
        v 1.000000 1.000000 -1.000000
        vt 0.748573 0.750412
        vt 0.749279 0.501284
        vt 0.999110 0.501077
        vt 0.999455 0.750380
        vn 0.000000 0.000000 -1.000000
        vn -1.000000 -0.000000 -0.000000
        vn -0.000000 -0.000000 1.000000
        f 5/2/1 1/2/1 4/3/1
        f 5/1/1 4/3/1 2/4/1

    The first character of the line denotes the type of input:
    ::
        - v is a vertex
        - vt is the texture coordinate of one vertex
        - vn is the normal of one vertex
        - f is a face

    Faces are interpreted as follows:
    ::
        5/2/1 describes the first vertex of the first triangle
        - 5: index of vertex [1.000000 1.000000 -1.000000]
        - 2: index of texture coordinate [0.749279 0.501284]
        - 1: index of normal [0.000000 0.000000 -1.000000]

    If there are faces with more than 3 vertices
    they are subdivided into triangles. Polygonal faces are assumed to have
    vertices ordered counter-clockwise so the (right-handed) normal points
    out of the screen e.g. a proper rectangular face would be specified like this:
    ::
        0_________1
        |         |
        |         |
        3 ________2

    The face would be split into two triangles: (0, 2, 1) and (0, 3, 2),
    both of which are also oriented counter-clockwise and have normals
    pointing out of the screen.

    Args:
        f: A file-like object (with methods read, readline, tell, and seek),
           a pathlib path or a string containing a file name.
        load_textures: Boolean indicating whether material files are loaded
        create_texture_atlas: Bool, If True a per face texture map is created and
            a tensor `texture_atlas` is also returned in `aux`.
        texture_atlas_size: Int specifying the resolution of the texture map per face
            when `create_texture_atlas=True`. A (texture_size, texture_size, 3)
            map is created per face.
        texture_wrap: string, one of ["repeat", "clamp"]. This applies when computing
            the texture atlas.
            If `texture_mode="repeat"`, for uv values outside the range [0, 1] the integer part
            is ignored and a repeating pattern is formed.
            If `texture_mode="clamp"` the values are clamped to the range [0, 1].
            If None, then there is no transformation of the texture values.
        device: Device (as str or torch.device) on which to return the new tensors.
        path_manager: optionally a PathManager object to interpret paths.

    Returns:
        6-element tuple containing

        - **verts**: FloatTensor of shape (V, 3).
        - **faces**: NamedTuple with fields:
            - verts_idx: LongTensor of vertex indices, shape (F, 3).
            - normals_idx: (optional) LongTensor of normal indices, shape (F, 3).
            - textures_idx: (optional) LongTensor of texture indices, shape (F, 3).
              This can be used to index into verts_uvs.
            - materials_idx: (optional) List of indices indicating which
              material the texture is derived from for each face.
              If there is no material for a face, the index is -1.
              This can be used to retrieve the corresponding values
              in material_colors/texture_images after they have been
              converted to tensors or Materials/Textures data
              structures - see textures.py and materials.py for
              more info.
        - **aux**: NamedTuple with fields:
            - normals: FloatTensor of shape (N, 3)
            - verts_uvs: FloatTensor of shape (T, 2), giving the uv coordinate per
              vertex. If a vertex is shared between two faces, it can have
              a different uv value for each instance. Therefore it is
              possible that the number of verts_uvs is greater than
              num verts i.e. T > V.
              vertex.
            - material_colors: if `load_textures=True` and the material has associated
              properties this will be a dict of material names and properties of the form:

              .. code-block:: python

                  {
                      material_name_1:  {
                          "ambient_color": tensor of shape (1, 3),
                          "diffuse_color": tensor of shape (1, 3),
                          "specular_color": tensor of shape (1, 3),
                          "shininess": tensor of shape (1)
                      },
                      material_name_2: {},
                      ...
                  }

              If a material does not have any properties it will have an
              empty dict. If `load_textures=False`, `material_colors` will None.

            - texture_images: if `load_textures=True` and the material has a texture map,
              this will be a dict of the form:

              .. code-block:: python

                  {
                      material_name_1: (H, W, 3) image,
                      ...
                  }
              If `load_textures=False`, `texture_images` will None.
            - texture_atlas: if `load_textures=True` and `create_texture_atlas=True`,
              this will be a FloatTensor of the form: (F, texture_size, textures_size, 3)
              If the material does not have a texture map, then all faces
              will have a uniform white texture.  Otherwise `texture_atlas` will be
              None.
    """
    data_dir = "./"
    if isinstance(f, (str, bytes, Path)):
        # pyre-fixme[6]: For 1st argument expected `PathLike[Variable[AnyStr <:
        #  [str, bytes]]]` but got `Union[Path, bytes, str]`.
        data_dir = os.path.dirname(f)
    if path_manager is None:
        path_manager = PathManager()
    with _open_file(f, path_manager, "r") as f:
        return _load_obj(
            f,
            data_dir=data_dir,
            load_textures=load_textures,
            create_texture_atlas=create_texture_atlas,
            texture_atlas_size=texture_atlas_size,
            texture_wrap=texture_wrap,
            path_manager=path_manager,
            device=device,
        )

def save_obj(
    f: PathOrStr,
    verts,
    faces,
    decimal_places: Optional[int] = None,
    path_manager: Optional[PathManager] = None,
    *,
    normals: Optional[torch.Tensor] = None,
    faces_normals_idx: Optional[torch.Tensor] = None,
    verts_uvs: Optional[torch.Tensor] = None,
    faces_uvs: Optional[torch.Tensor] = None,
    texture_map: Optional[torch.Tensor] = None,
) -> None:
    """
    Save a mesh to an .obj file.

    Args:
        f: File (str or path) to which the mesh should be written.
        verts: FloatTensor of shape (V, 3) giving vertex coordinates.
        faces: LongTensor of shape (F, 3) giving faces.
        decimal_places: Number of decimal places for saving.
        path_manager: Optional PathManager for interpreting f if
            it is a str.
        normals: FloatTensor of shape (V, 3) giving normals for faces_normals_idx
            to index into.
        faces_normals_idx: LongTensor of shape (F, 3) giving the index into
            normals for each vertex in the face.
        verts_uvs: FloatTensor of shape (V, 2) giving the uv coordinate per vertex.
        faces_uvs: LongTensor of shape (F, 3) giving the index into verts_uvs for
            each vertex in the face.
        texture_map: FloatTensor of shape (H, W, 3) representing the texture map
            for the mesh which will be saved as an image. The values are expected
            to be in the range [0, 1],
    """
    if len(verts) and (verts.dim() != 2 or verts.size(1) != 3):
        message = "'verts' should either be empty or of shape (num_verts, 3)."
        raise ValueError(message)

    if len(faces) and (faces.dim() != 2 or faces.size(1) != 3):
        message = "'faces' should either be empty or of shape (num_faces, 3)."
        raise ValueError(message)

    if (normals is None) != (faces_normals_idx is None):
        message = "'normals' and 'faces_normals_idx' must both be None or neither."
        raise ValueError(message)

    if faces_normals_idx is not None and (
        faces_normals_idx.dim() != 2 or faces_normals_idx.size(1) != 3
    ):
        message = (
            "'faces_normals_idx' should either be empty or of shape (num_faces, 3)."
        )
        raise ValueError(message)

    if normals is not None and (normals.dim() != 2 or normals.size(1) != 3):
        message = "'normals' should either be empty or of shape (num_verts, 3)."
        raise ValueError(message)

    if faces_uvs is not None and (faces_uvs.dim() != 2 or faces_uvs.size(1) != 3):
        message = "'faces_uvs' should either be empty or of shape (num_faces, 3)."
        raise ValueError(message)

    if verts_uvs is not None and (verts_uvs.dim() != 2 or verts_uvs.size(1) != 2):
        message = "'verts_uvs' should either be empty or of shape (num_verts, 2)."
        raise ValueError(message)

    if texture_map is not None and (texture_map.dim() != 3 or texture_map.size(2) != 3):
        message = "'texture_map' should either be empty or of shape (H, W, 3)."
        raise ValueError(message)

    if path_manager is None:
        path_manager = PathManager()

    save_texture = all(t is not None for t in [faces_uvs, verts_uvs, texture_map])
    output_path = Path(f)

    # Save the .obj file
    with _open_file(f, path_manager, "w") as f:
        if save_texture:
            # Add the header required for the texture info to be loaded correctly
            obj_header = "\nmtllib {0}.mtl\nusemtl mesh\n\n".format(output_path.stem)
            f.write(obj_header)
        _save(
            f,
            verts,
            faces,
            decimal_places,
            normals=normals,
            faces_normals_idx=faces_normals_idx,
            verts_uvs=verts_uvs,
            faces_uvs=faces_uvs,
            save_texture=save_texture,
            save_normals=normals is not None,
        )

    # Save the .mtl and .png files associated with the texture
    if save_texture:
        image_path = output_path.with_suffix(".png")
        mtl_path = output_path.with_suffix(".mtl")
        if isinstance(f, str):
            # Back to str for iopath interpretation.
            image_path = str(image_path)
            mtl_path = str(mtl_path)

        # Save texture map to output folder
        # pyre-fixme[16] # undefined attribute cpu
        texture_map = texture_map.detach().cpu() * 255.0
        image = Image.fromarray(texture_map.numpy().astype(np.uint8))
        with _open_file(image_path, path_manager, "wb") as im_f:
            image.save(im_f)

        # Create .mtl file with the material name and texture map filename
        # TODO: enable material properties to also be saved.
        with _open_file(mtl_path, path_manager, "w") as f_mtl:
            lines = f"newmtl mesh\n" f"map_Kd {output_path.stem}.png\n"
            f_mtl.write(lines)

