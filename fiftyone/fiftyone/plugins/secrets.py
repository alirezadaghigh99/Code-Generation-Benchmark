class PluginSecretsResolver:
    """Injects secrets from environmental variables into the execution
    context."""

    _instance = None
    _registered_secrets = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PluginSecretsResolver, cls).__new__(cls)
            cls._instance.client = _get_secrets_client()
        return cls._instance

    def register_operator(
        self, operator_uri: str, required_secrets: typing.List[str]
    ) -> None:
        self._registered_secrets[operator_uri] = required_secrets

    def client(self) -> fois.ISecretProvider:
        if not self._instance:
            self._instance = self.__new__(self.__class__)
        return self._instance.client

    async def get_secret(
        self, key: str, operator_uri: str, **kwargs
    ) -> Optional[fois.ISecret]:
        """
        Get the value of a secret.

        Args:
            key (str): unique secret identifier
            kwargs: additional keyword arguments to pass to the secrets
            client for authentication if required
        """
        # pylint: disable=no-member

        secret_requirements = self._registered_secrets.get(operator_uri, None)

        if not secret_requirements:
            logging.error(
                f"Cannot resolve secrets for unregistered "
                f"operator`{operator_uri}` "
            )
            return None
        if key not in secret_requirements:
            logging.error(
                f"Cannot resolve secret {key} because it is not "
                f"included in the plugins definition "
            )
            return None
        resolved_secret = await self.client.get(key, **kwargs)
        return resolved_secret

    def get_secret_sync(
        self, key: str, operator_uri: str, **kwargs
    ) -> Optional[fois.ISecret]:
        """
        Get the value of a secret.

        Args:
            key (str): unique secret identifier
            kwargs: additional keyword arguments to pass to the secrets
            client for authentication if required
        """
        # pylint: disable=no-member

        secret_requirements = self._registered_secrets.get(operator_uri, None)
        if not secret_requirements:
            logging.error(
                f"Cannot resolve secrets for unregistered "
                f"operator`{operator_uri}` "
            )
            return None
        if key not in secret_requirements:
            logging.error(
                f"Cannot resolve secret {key} because it is not "
                f"included in the plugins definition "
            )
            return None
        return self.client.get_sync(key, **kwargs)

