def check_equal(
        cls,
        test: Union[TensorType, List[TensorType]],
        reference: Union[TensorType, List[TensorType]],
        rtol: float = 1e-1,
        atol=0,
    ):
        cls._check_assertion(test, reference, lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol, atol=atol))

def create_venv_with_nncf(tmp_path: Path, package_type: str, venv_type: str, backends: Set[str] = None):
    venv_path = tmp_path / "venv"
    venv_path.mkdir()

    python_executable_with_venv = get_python_executable_with_venv(venv_path)
    pip_with_venv = get_pip_executable_with_venv(venv_path)

    version_string = f"{sys.version_info[0]}.{sys.version_info[1]}"

    if venv_type == "virtualenv":
        virtualenv = Path(sys.executable).parent / "virtualenv"
        subprocess.check_call(f"{virtualenv} -ppython{version_string} {venv_path}", shell=True)
    elif venv_type == "venv":
        subprocess.check_call(f"{sys.executable} -m venv {venv_path}", shell=True)

    subprocess.check_call(f"{pip_with_venv} install --upgrade pip", shell=True)
    subprocess.check_call(f"{pip_with_venv} install --upgrade wheel setuptools", shell=True)

    if package_type in ["build_s", "build_w"]:
        subprocess.check_call(f"{pip_with_venv} install build", shell=True)

    run_path = tmp_path / "run"
    run_path.mkdir()

    if package_type == "pip_pypi":
        run_cmd_line = f"{pip_with_venv} install nncf"
    elif package_type == "pip_local":
        run_cmd_line = f"{pip_with_venv} install {PROJECT_ROOT}"
    elif package_type == "pip_e_local":
        run_cmd_line = f"{pip_with_venv} install -e {PROJECT_ROOT}"
    elif package_type == "pip_git_develop":
        run_cmd_line = f"{pip_with_venv} install git+{GITHUB_REPO_URL}@develop#egg=nncf"
    elif package_type == "build_s":
        run_cmd_line = f"{python_executable_with_venv} -m build -n -s"
    elif package_type == "build_w":
        run_cmd_line = f"{python_executable_with_venv} -m build -n -w"
    else:
        raise nncf.ValidationError(f"Invalid package type: {package_type}")

    subprocess.run(run_cmd_line, check=True, shell=True, cwd=PROJECT_ROOT)
    if backends:
        # Install backend specific packages with according version from constraints.txt
        packages = [item for b in backends for item in MAP_BACKEND_PACKAGES[b]]
        extra_reqs = " ".join(packages)
        subprocess.run(
            f"{pip_with_venv} install {extra_reqs} -c {PROJECT_ROOT}/constraints.txt",
            check=True,
            shell=True,
            cwd=PROJECT_ROOT,
        )

    return venv_path

