class Diagnostic(infra.Diagnostic):
    logger: logging.Logger = dataclasses.field(init=False, default=diagnostic_logger)

    def log(self, level: int, message: str, *args, **kwargs) -> None:
        if self.logger.isEnabledFor(level):
            formatted_message = message % args
            if is_onnx_diagnostics_log_artifact_enabled():
                # Only log to terminal if artifact is enabled.
                # See [NOTE: `dynamo_export` diagnostics logging] for details.
                self.logger.log(level, formatted_message, **kwargs)

            self.additional_messages.append(formatted_message)

