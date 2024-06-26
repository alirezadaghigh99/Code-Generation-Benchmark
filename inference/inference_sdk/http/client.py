    def list_loaded_models(self) -> RegisteredModels:
        self.__ensure_v1_client_mode()
        response = requests.get(f"{self.__api_url}/model/registry")
        response.raise_for_status()
        response_payload = response.json()
        return RegisteredModels.from_dict(response_payload)