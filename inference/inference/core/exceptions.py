class InputImageLoadError(Exception):

    def __init__(self, message: str, public_message: str):
        super().__init__(message)
        self._public_message = public_message

    def get_public_error_details(self) -> str:
        return self._public_message