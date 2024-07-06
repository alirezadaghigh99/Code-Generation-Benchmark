def get_default_qconfig_mapping(backend="x86", version=0) -> QConfigMapping:
    """
    Return the default QConfigMapping for post training quantization.

    Args:
      * ``backend`` (str) : the quantization backend for the default qconfig mapping, should be
         one of ["x86" (default), "fbgemm", "qnnpack", "onednn"]
      * ``version`` (int) : the version for the default qconfig mapping
    """
    # TODO: add assert for backend choices
    return _get_default_qconfig_mapping(False, backend, version)

