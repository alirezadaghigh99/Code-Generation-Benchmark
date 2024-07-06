def extract_archive(src: str, dst: str | None = None) -> None:
    """Extract an archive.

    Args:
        src: file to be extracted
        dst: directory to extract to (defaults to dirname of ``src``)

    Raises:
        RuntimeError: if src file has unknown archival/compression scheme
    """
    if dst is None:
        dst = os.path.dirname(src)

    suffix_and_extractor: list[tuple[str | tuple[str, ...], Any]] = [
        ('.rar', _rarfile.RarFile),
        (
            ('.tar', '.tar.gz', '.tar.bz2', '.tar.xz', '.tgz', '.tbz2', '.tbz', '.txz'),
            tarfile.open,
        ),
        ('.zip', _zipfile.ZipFile),
    ]

    for suffix, extractor in suffix_and_extractor:
        if src.endswith(suffix):
            with extractor(src, 'r') as f:
                f.extractall(dst)
            return

    suffix_and_decompressor: list[tuple[str, Any]] = [
        ('.bz2', bz2.open),
        ('.gz', gzip.open),
        ('.xz', lzma.open),
    ]

    for suffix, decompressor in suffix_and_decompressor:
        if src.endswith(suffix):
            dst = os.path.join(dst, os.path.basename(src).replace(suffix, ''))
            with decompressor(src, 'rb') as sf, open(dst, 'wb') as df:
                df.write(sf.read())
            return

    raise RuntimeError('src file has unknown archival/compression scheme')

def working_dir(dirname: str, create: bool = False) -> Iterator[None]:
    """Context manager for changing directories.

    Args:
        dirname: directory to temporarily change to
        create: if True, create the destination directory
    """
    if create:
        os.makedirs(dirname, exist_ok=True)

    cwd = os.getcwd()
    os.chdir(dirname)

    try:
        yield
    finally:
        os.chdir(cwd)

def lazy_import(name: str) -> Any:
    """Lazy import of *name*.

    Args:
        name: Name of module to import.

    Returns:
        Module import.

    Raises:
        DependencyNotFoundError: If *name* is not installed.

    .. versionadded:: 0.6
    """
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        # Map from import name to package name on PyPI
        name = name.split('.')[0].replace('_', '-')
        module_to_pypi: dict[str, str] = collections.defaultdict(lambda: name)
        module_to_pypi |= {'cv2': 'opencv-python', 'skimage': 'scikit-image'}
        name = module_to_pypi[name]
        msg = f"""\
{name} is not installed and is required to use this dataset. Either run:

$ pip install {name}

to install just this dependency, or:

$ pip install torchgeo[datasets]

to install all optional dataset dependencies."""
        raise DependencyNotFoundError(msg) from None

def which(name: str) -> Executable:
    """Search for executable *name*.

    Args:
        name: Name of executable to search for.

    Returns:
        Callable executable instance.

    Raises:
        DependencyNotFoundError: If *name* is not installed.

    .. versionadded:: 0.6
    """
    if shutil.which(name):
        return Executable(name)
    else:
        msg = f'{name} is not installed and is required to use this dataset.'
        raise DependencyNotFoundError(msg) from None

