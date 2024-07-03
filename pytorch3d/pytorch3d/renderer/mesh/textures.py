class TexturesVertex(TexturesBase):
    def __init__(
        self,
        verts_features: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
    ) -> None:
        """
        Batched texture representation where each vertex in a mesh
        has a C dimensional feature vector.

        Args:
            verts_features: list of (Vi, C) or (N, V, C) tensor giving a feature
                vector with arbitrary dimensions for each vertex.
        """
        if isinstance(verts_features, (tuple, list)):
            correct_shape = all(
                (torch.is_tensor(v) and v.ndim == 2) for v in verts_features
            )
            if not correct_shape:
                raise ValueError(
                    "Expected verts_features to be a list of tensors of shape (V, C)."
                )

            self._verts_features_list = verts_features
            self._verts_features_padded = None
            self.device = torch.device("cpu")

            # These values may be overridden when textures is
            # passed into the Meshes constructor. For more details
            # refer to the __init__ of Meshes.
            self._N = len(verts_features)
            self._num_verts_per_mesh = [len(fv) for fv in verts_features]

            if self._N > 0:
                self.device = verts_features[0].device

        elif torch.is_tensor(verts_features):
            if verts_features.ndim != 3:
                msg = "Expected verts_features to be of shape (N, V, C); got %r"
                raise ValueError(msg % repr(verts_features.shape))
            self._verts_features_padded = verts_features
            self._verts_features_list = None
            self.device = verts_features.device

            # These values may be overridden when textures is
            # passed into the Meshes constructor. For more details
            # refer to the __init__ of Meshes.
            self._N = len(verts_features)
            max_F = verts_features.shape[1]
            self._num_verts_per_mesh = [max_F] * self._N
        else:
            raise ValueError("verts_features must be a tensor or list of tensors")

        # This is set inside the Meshes object when textures is
        # passed into the Meshes constructor. For more details
        # refer to the __init__ of Meshes.
        self.valid = torch.ones((self._N,), dtype=torch.bool, device=self.device)

    def clone(self) -> "TexturesVertex":
        tex = self.__class__(self.verts_features_padded().clone())
        if self._verts_features_list is not None:
            tex._verts_features_list = [f.clone() for f in self._verts_features_list]
        tex._num_verts_per_mesh = self._num_verts_per_mesh.copy()
        tex.valid = self.valid.clone()
        return tex

    def detach(self) -> "TexturesVertex":
        tex = self.__class__(self.verts_features_padded().detach())
        if self._verts_features_list is not None:
            tex._verts_features_list = [f.detach() for f in self._verts_features_list]
        tex._num_verts_per_mesh = self._num_verts_per_mesh.copy()
        tex.valid = self.valid.detach()
        return tex

    def __getitem__(self, index) -> "TexturesVertex":
        props = ["verts_features_list", "_num_verts_per_mesh"]
        new_props = self._getitem(index, props)
        verts_features = new_props["verts_features_list"]
        if isinstance(verts_features, list):
            # Handle the case of an empty list
            if len(verts_features) == 0:
                verts_features = torch.empty(
                    size=(0, 0, 3),
                    dtype=torch.float32,
                    device=self.verts_features_padded().device,
                )
            new_tex = self.__class__(verts_features=verts_features)
        elif torch.is_tensor(verts_features):
            new_tex = self.__class__(verts_features=[verts_features])
        else:
            raise ValueError("Not all values are provided in the correct format")
        new_tex._num_verts_per_mesh = new_props["_num_verts_per_mesh"]
        return new_tex

    def verts_features_padded(self) -> torch.Tensor:
        if self._verts_features_padded is None:
            if self.isempty():
                self._verts_features_padded = torch.zeros(
                    (self._N, 0, 3, 0), dtype=torch.float32, device=self.device
                )
            else:
                self._verts_features_padded = list_to_padded(
                    self._verts_features_list, pad_value=0.0
                )
        return self._verts_features_padded

    def verts_features_list(self) -> List[torch.Tensor]:
        if self._verts_features_list is None:
            if self.isempty():
                self._verts_features_list = [
                    torch.empty((0, 3), dtype=torch.float32, device=self.device)
                ] * self._N
            else:
                self._verts_features_list = padded_to_list(
                    self._verts_features_padded, split_size=self._num_verts_per_mesh
                )
        return self._verts_features_list

    def verts_features_packed(self) -> torch.Tensor:
        if self.isempty():
            return torch.zeros((self._N, 3, 0), dtype=torch.float32, device=self.device)
        verts_features_list = self.verts_features_list()
        return list_to_packed(verts_features_list)[0]

    def extend(self, N: int) -> "TexturesVertex":
        new_props = self._extend(N, ["verts_features_padded", "_num_verts_per_mesh"])
        new_tex = self.__class__(verts_features=new_props["verts_features_padded"])
        new_tex._num_verts_per_mesh = new_props["_num_verts_per_mesh"]
        return new_tex

    # pyre-fixme[14]: `sample_textures` overrides method defined in `TexturesBase`
    #  inconsistently.
    def sample_textures(self, fragments, faces_packed=None) -> torch.Tensor:
        """
        Determine the color for each rasterized face. Interpolate the colors for
        vertices which form the face using the barycentric coordinates.
        Args:
            fragments:
                The outputs of rasterization. From this we use

                - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
                of the faces (in the packed representation) which
                overlap each pixel in the image.
                - barycentric_coords: FloatTensor of shape (N, H, W, K, 3) specifying
                the barycentric coordinates of each pixel
                relative to the faces (in the packed
                representation) which overlap the pixel.

        Returns:
            texels: An texture per pixel of shape (N, H, W, K, C).
            There will be one C dimensional value for each element in
            fragments.pix_to_face.
        """
        verts_features_packed = self.verts_features_packed()
        faces_verts_features = verts_features_packed[faces_packed]

        texels = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_verts_features
        )
        return texels

    def submeshes(
        self,
        vertex_ids_list: List[List[torch.LongTensor]],
        faces_ids_list: List[List[torch.LongTensor]],
    ) -> "TexturesVertex":
        """
        Extract a sub-texture for use in a submesh.

        If the meshes batch corresponding to this TexturesVertex contains
        `n = len(vertex_ids_list)` meshes, then self.verts_features_list()
        will be of length n. After submeshing, we obtain a batch of
        `k = sum(len(v) for v in vertex_ids_list` submeshes (see Meshes.submeshes). This
        function creates a corresponding TexturesVertex object with `verts_features_list`
        of length `k`.

        Args:
            vertex_ids_list: A list of length equal to self.verts_features_list. Each
                element is a LongTensor listing the vertices that the submesh keeps in
                each respective mesh.

            face_ids_list: Not used when submeshing TexturesVertex.

        Returns:
            A TexturesVertex in which verts_features_list has length
            sum(len(vertices) for vertices in vertex_ids_list). Each element contains
            vertex features corresponding to the subset of vertices in that submesh.
        """
        if len(vertex_ids_list) != len(self.verts_features_list()):
            raise IndexError(
                "verts_features_list must be of " "the same length as vertex_ids_list."
            )

        sub_features = []
        for vertex_ids, features in zip(vertex_ids_list, self.verts_features_list()):
            for vertex_ids_submesh in vertex_ids:
                sub_features.append(features[vertex_ids_submesh])

        return self.__class__(sub_features)

    def faces_verts_textures_packed(self, faces_packed=None) -> torch.Tensor:
        """
        Samples texture from each vertex and for each face in the mesh.
        For N meshes with {Fi} number of faces, it returns a
        tensor of shape sum(Fi)x3xC (C = 3 for RGB).
        You can use the utils function in structures.utils to convert the
        packed representation to a list or padded.
        """
        verts_features_packed = self.verts_features_packed()
        faces_verts_features = verts_features_packed[faces_packed]
        return faces_verts_features

    def join_batch(self, textures: List["TexturesVertex"]) -> "TexturesVertex":
        """
        Join the list of textures given by `textures` to
        self to create a batch of textures. Return a new
        TexturesVertex object with the combined textures.

        Args:
            textures: List of TexturesVertex objects

        Returns:
            new_tex: TexturesVertex object with the combined
            textures from self and the list `textures`.
        """
        tex_types_same = all(isinstance(tex, TexturesVertex) for tex in textures)
        if not tex_types_same:
            raise ValueError("All textures must be of type TexturesVertex.")

        verts_features_list = []
        verts_features_list += self.verts_features_list()
        num_verts_per_mesh = self._num_verts_per_mesh.copy()
        for tex in textures:
            verts_features_list += tex.verts_features_list()
            num_verts_per_mesh += tex._num_verts_per_mesh

        new_tex = self.__class__(verts_features=verts_features_list)
        new_tex._num_verts_per_mesh = num_verts_per_mesh
        return new_tex

    def join_scene(self) -> "TexturesVertex":
        """
        Return a new TexturesVertex amalgamating the batch.
        """
        return self.__class__(verts_features=[torch.cat(self.verts_features_list())])

    def check_shapes(
        self, batch_size: int, max_num_verts: int, max_num_faces: int
    ) -> bool:
        """
        Check if the dimensions of the verts features match that of the mesh verts
        """
        # (N, V) should be the same
        return self.verts_features_padded().shape[:-1] == (batch_size, max_num_verts)