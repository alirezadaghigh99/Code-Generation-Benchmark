def get_test_asset_path(*args):
    cur_dir = osp.dirname(__file__)
    return osp.abspath(osp.join(cur_dir, "..", "assets", *args))

