class CommandContext(BaseModel):
    request_id: Optional[str] = Field(
        description="Server-side request ID", default=None
    )
    pipeline_id: Optional[str] = Field(
        description="Identifier of pipeline connected to operation", default=None
    )

class CommandResponse(BaseModel):
    status: str = Field(description="Operation status")
    context: CommandContext = Field(description="Context of the command.")

class ListPipelinesResponse(CommandResponse):
    pipelines: List[str] = Field(description="List IDs of active pipelines")

class InferencePipelineStatusResponse(CommandResponse):
    report: dict

