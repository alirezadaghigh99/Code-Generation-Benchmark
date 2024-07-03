def execute_notebook(
    input_notebook, output_notebook, parameters={}, kernel_name="python3", timeout=2200
):
    """Execute a notebook while passing parameters to it.

    Note:
        Ensure your Jupyter Notebook is set up with parameters that can be
        modified and read. Use Markdown cells to specify parameters that need
        modification and code cells to set parameters that need to be read.

    Args:
        input_notebook (str): Path to the input notebook.
        output_notebook (str): Path to the output notebook
        parameters (dict): Dictionary of parameters to pass to the notebook.
        kernel_name (str): Kernel name.
        timeout (int): Timeout (in seconds) for each cell to execute.
    """

    # Load the Jupyter Notebook
    with open(input_notebook, "r") as notebook_file:
        notebook_content = nbformat.read(notebook_file, as_version=4)

    # Search for and replace parameter values in code cells
    for cell in notebook_content.cells:
        if (
            "tags" in cell.metadata
            and "parameters" in cell.metadata["tags"]
            and cell.cell_type == "code"
        ):
            # Update the cell's source within notebook_content
            cell.source = _update_parameters(cell.source, parameters)

    # Create an execution preprocessor
    execute_preprocessor = ExecutePreprocessor(timeout=timeout, kernel_name=kernel_name)

    # Execute the notebook
    executed_notebook, _ = execute_preprocessor.preprocess(
        notebook_content, {"metadata": {"path": "./"}}
    )

    # Save the executed notebook
    with open(output_notebook, "w", encoding="utf-8") as executed_notebook_file:
        nbformat.write(executed_notebook, executed_notebook_file)def execute_notebook(
    input_notebook, output_notebook, parameters={}, kernel_name="python3", timeout=2200
):
    """Execute a notebook while passing parameters to it.

    Note:
        Ensure your Jupyter Notebook is set up with parameters that can be
        modified and read. Use Markdown cells to specify parameters that need
        modification and code cells to set parameters that need to be read.

    Args:
        input_notebook (str): Path to the input notebook.
        output_notebook (str): Path to the output notebook
        parameters (dict): Dictionary of parameters to pass to the notebook.
        kernel_name (str): Kernel name.
        timeout (int): Timeout (in seconds) for each cell to execute.
    """

    # Load the Jupyter Notebook
    with open(input_notebook, "r") as notebook_file:
        notebook_content = nbformat.read(notebook_file, as_version=4)

    # Search for and replace parameter values in code cells
    for cell in notebook_content.cells:
        if (
            "tags" in cell.metadata
            and "parameters" in cell.metadata["tags"]
            and cell.cell_type == "code"
        ):
            # Update the cell's source within notebook_content
            cell.source = _update_parameters(cell.source, parameters)

    # Create an execution preprocessor
    execute_preprocessor = ExecutePreprocessor(timeout=timeout, kernel_name=kernel_name)

    # Execute the notebook
    executed_notebook, _ = execute_preprocessor.preprocess(
        notebook_content, {"metadata": {"path": "./"}}
    )

    # Save the executed notebook
    with open(output_notebook, "w", encoding="utf-8") as executed_notebook_file:
        nbformat.write(executed_notebook, executed_notebook_file)