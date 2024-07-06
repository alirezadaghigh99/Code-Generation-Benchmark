def box_corners_from_param(box_size, heading_angle, center):
    """Generates box corners from a parameterised box.
    box_size is array(size_x,size_y,size_z), heading_angle is radius clockwise from pos x axis, center is xyz of box center
        output (8,3) array for 3D box corners
    """
    R = euler_angles_to_rotation_matrix(torch.tensor([0.0, 0.0, float(heading_angle)]))
    if torch.is_tensor(box_size):
        box_size = box_size.float()
    l, w, h = box_size
    x_corners = torch.tensor([-l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2])
    y_corners = torch.tensor([-w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2])
    z_corners = torch.tensor([-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2])
    corners_3d = R @ torch.stack([x_corners, y_corners, z_corners])
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = corners_3d.T
    return corners_3d

