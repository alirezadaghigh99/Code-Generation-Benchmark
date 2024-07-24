class MockedApiWorkflowClient(ApiWorkflowClient):
    embeddings_filename_base = "img"
    n_embedding_rows_on_server = N_FILES_ON_SERVER

    def __init__(self, *args, **kwargs):
        ApiWorkflowClient.__init__(self, *args, **kwargs)

        self._selection_api = MockedSamplingsApi(api_client=self.api_client)
        self._jobs_api = MockedJobsApi(api_client=self.api_client)
        self._tags_api = MockedTagsApi(api_client=self.api_client)
        self._embeddings_api = MockedEmbeddingsApi(api_client=self.api_client)
        self._samples_api = MockedSamplesApi(api_client=self.api_client)
        self._mappings_api = MockedMappingsApi(
            api_client=self.api_client, samples_api=self._samples_api
        )
        self._scores_api = MockedScoresApi(api_client=self.api_client)
        self._datasets_api = MockedDatasetsApi(api_client=self.api_client)
        self._datasources_api = MockedDatasourcesApi(api_client=self.api_client)
        self._quota_api = MockedQuotaApi(api_client=self.api_client)
        self._compute_worker_api = MockedComputeWorkerApi(api_client=self.api_client)
        self._collaboration_api = MockedAPICollaboration(api_client=self.api_client)

        lightly.api.api_workflow_client.requests.put = mocked_request_put

        self.wait_time_till_next_poll = 0.001  # for api_workflow_selection

    def upload_file_with_signed_url(
        self,
        file: IOBase,
        signed_write_url: str,
        max_backoff: int = 32,
        max_retries: int = 5,
        headers: Dict = None,
        session: Optional[requests.Session] = None,
    ) -> Response:
        res = Response()
        return res

    def _get_csv_reader_from_read_url(self, read_url: str):
        n_rows: int = self.n_embedding_rows_on_server
        n_dims: int = self.n_dims_embeddings_on_server

        rows_csv = [
            ["filenames"] + [f"embedding_{i}" for i in range(n_dims)] + ["labels"]
        ]
        for i in range(n_rows):
            row = [f"{self.embeddings_filename_base}_{i}.jpg"]
            for _ in range(n_dims):
                row.append(np.random.uniform(0, 1))
            row.append(i)
            rows_csv.append(row)

        # save the csv rows in a temporary in-memory string file
        # using a csv writer and then read them as bytes
        f = tempfile.SpooledTemporaryFile(mode="rw")
        writer = csv.writer(f)
        writer.writerows(rows_csv)
        f.seek(0)
        buffer = io.StringIO(f.read())
        reader = csv.reader(buffer)

        return reader

