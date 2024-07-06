def _parse_ground_truth_data(data):
    gt = dict()
    for i, arr in enumerate(data):
        temp = []
        for row in arr:
            if np.isnan(row[:, :2]).all():
                continue
            ass = Assembly.from_array(row)
            temp.append(ass)
        if not temp:
            continue
        gt[i] = temp
    return gt

