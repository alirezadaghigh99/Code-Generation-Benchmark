def load_dataset(id=None, name=None):
    """Loads the dataset from the database by its unique id or name. Throws
    an error if neither id nor name is provided.

    Args:
        id (None): the unique id of the dataset
        name (None): the name of the dataset

    Returns:
        a :class:`fiftyone.core.dataset.Dataset`
    """
    import fiftyone.core.odm as foo
    import fiftyone.core.dataset as fod

    if name:
        return fod.load_dataset(name)

    if not id:
        raise ValueError("Must provide either id or name")

    db = foo.get_db_conn()
    try:
        uid = ObjectId(id)
    except:
        # Although _id is an ObjectId by default, it's possible to set it to
        # something else
        uid = id

    res = db.datasets.find_one({"_id": uid}, {"name": True})
    if not res:
        raise ValueError(f"Dataset with _id={uid} does not exist")
    return fod.load_dataset(res.get("name"))

