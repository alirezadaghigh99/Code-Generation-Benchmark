def right_jacobian(vec: Tensor) -> Tensor:
        """Computes the right Jacobian of So3.

        Args:
            vec: the input point of shape :math:`(B, 3)`.

        Example:
            >>> vec = torch.tensor([1., 2., 3.])
            >>> So3.right_jacobian(vec)
            tensor([[-0.0687,  0.5556, -0.0141],
                    [-0.2267,  0.1779,  0.6236],
                    [ 0.5074,  0.3629,  0.5890]])
        """
        # KORNIA_CHECK_SHAPE(vec, ["B", "3"])  # FIXME: resolve shape bugs. @edgarriba
        R_skew = vector_to_skew_symmetric_matrix(vec)
        theta = vec.norm(dim=-1, keepdim=True)[..., None]
        I = eye(3, device=vec.device, dtype=vec.dtype)  # noqa: E741
        Jr = I - ((1 - theta.cos()) / theta**2) * R_skew + ((theta - theta.sin()) / theta**3) * (R_skew @ R_skew)
        return Jr

def left_jacobian(vec: Tensor) -> Tensor:
        """Computes the left Jacobian of So3.

        Args:
            vec: the input point of shape :math:`(B, 3)`.

        Example:
            >>> vec = torch.tensor([1., 2., 3.])
            >>> So3.left_jacobian(vec)
            tensor([[-0.0687, -0.2267,  0.5074],
                    [ 0.5556,  0.1779,  0.3629],
                    [-0.0141,  0.6236,  0.5890]])
        """
        # KORNIA_CHECK_SHAPE(vec, ["B", "3"])  # FIXME: resolve shape bugs. @edgarriba
        R_skew = vector_to_skew_symmetric_matrix(vec)
        theta = vec.norm(dim=-1, keepdim=True)[..., None]
        I = eye(3, device=vec.device, dtype=vec.dtype)  # noqa: E741
        Jl = I + ((1 - theta.cos()) / theta**2) * R_skew + ((theta - theta.sin()) / theta**3) * (R_skew @ R_skew)
        return Jl

