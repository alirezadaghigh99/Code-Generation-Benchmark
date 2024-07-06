def list_datasets():
            return [
                name
                for name in fo.list_datasets()
                if name not in IGNORED_DATASET_NAMES
            ]

