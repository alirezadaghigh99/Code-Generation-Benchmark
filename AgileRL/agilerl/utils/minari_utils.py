def load_minari_dataset(dataset_id, accelerator=None, remote=False):
    if remote:
        if dataset_id not in list(minari.list_remote_datasets().keys()):
            raise KeyError(
                "Enter a valid remote Minari Dataset ID. check https://minari.farama.org/ for more details."
            )

    file_path = get_dataset_path(dataset_id)

    if not os.path.exists(file_path):
        if remote:
            if accelerator is not None:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    print("download dataset: ", dataset_id)
                    download_dataset(dataset_id)
                accelerator.wait_for_everyone()
            else:
                print("download dataset: ", dataset_id)
                download_dataset(dataset_id)
        else:
            raise FileNotFoundError(
                f"No local Dataset found for dataset id {dataset_id}. check https://minari.farama.org/ for more details on remote dataset. For loading a remote dataset assign remote=True"
            )

    minari_dataset = load_dataset(dataset_id)

    return minari_dataset

