def from_numpy(a):
    # If not numpy array, piggy back on e.g. tensor guards to check type
    return torch.as_tensor(a) if isinstance(a, (np.generic, np.ndarray)) else a