def import_bioimageio(source, outpath):
    """Import stardist model from bioimage.io format, https://github.com/bioimage-io/spec-bioimage-io.

    Load a model in bioimage.io format from the given `source` (e.g. path to zip file, URL)
    and convert it to a regular stardist model, which will be saved in the folder `outpath`.

    Parameters
    ----------
    source: str, Path
        bioimage.io resource (e.g. path, URL)
    outpath: str, Path
        folder to save the stardist model (must not exist previously)

    Returns
    -------
    StarDist2D or StarDist3D
        stardist model loaded from `outpath`

    """
    import shutil, uuid
    from csbdeep.utils import save_json
    from .models import StarDist2D, StarDist3D
    *_, bioimageio_core, _ = _import()

    outpath = Path(outpath)
    not outpath.exists() or _raise(FileExistsError(f"'{outpath}' already exists"))

    with tempfile.TemporaryDirectory() as _tmp_dir:
        tmp_dir = Path(_tmp_dir)
        # download the full model content to a temporary folder
        zip_path = tmp_dir / f"{str(uuid.uuid4())}.zip"
        bioimageio_core.export_resource_package(source, output_path=zip_path)
        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)
        zip_path.unlink()
        rdf_path = tmp_dir / "rdf.yaml"
        biomodel = bioimageio_core.load_resource_description(rdf_path)

        # read the stardist specific content
        'stardist' in biomodel.config or _raise(RuntimeError("bioimage.io model not compatible"))
        config = biomodel.config['stardist']['config']
        thresholds = biomodel.config['stardist']['thresholds']
        weights = biomodel.config['stardist']['weights']

        # make sure that the keras weights are in the attachments
        weights_file = None
        for f in biomodel.attachments.files:
            if f.name == weights and f.exists():
                weights_file = f
                break
        weights_file is not None or _raise(FileNotFoundError(f"couldn't find weights file '{weights}'"))

        # save the config and threshold to json, and weights to hdf5 to enable loading as stardist model
        # copy bioimageio files to separate sub-folder
        outpath.mkdir(parents=True)
        save_json(config, str(outpath / 'config.json'))
        save_json(thresholds, str(outpath / 'thresholds.json'))
        shutil.copy(str(weights_file), str(outpath / "weights_bioimageio.h5"))
        shutil.copytree(str(tmp_dir), str(outpath / "bioimageio"))

    model_class = (StarDist2D if config['n_dim'] == 2 else StarDist3D)
    model = model_class(None, outpath.name, basedir=str(outpath.parent))

    return model