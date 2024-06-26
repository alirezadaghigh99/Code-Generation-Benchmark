class CommandResponse(BaseModel):
    status: str = Field(description="Operation status")
    context: CommandContext = Field(description="Context of the command.")