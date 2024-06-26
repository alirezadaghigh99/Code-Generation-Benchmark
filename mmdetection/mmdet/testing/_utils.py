def demo_mm_proposals(image_shapes, num_proposals, device='cpu'):
    """Create a list of fake porposals.

    Args:
        image_shapes (list[tuple[int]]): Batch image shapes.
        num_proposals (int): The number of fake proposals.
    """
    rng = np.random.RandomState(0)

    results = []
    for img_shape in image_shapes:
        result = InstanceData()
        w, h = img_shape[1:]
        proposals = _rand_bboxes(rng, num_proposals, w, h)
        result.bboxes = torch.from_numpy(proposals).float()
        result.scores = torch.from_numpy(rng.rand(num_proposals)).float()
        result.labels = torch.zeros(num_proposals).long()
        results.append(result.to(device))
    return results