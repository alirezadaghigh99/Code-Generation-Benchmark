class FlowControl(BaseModel):
    mode: Literal["pass", "terminate_branch", "select_step"]
    context: Optional[str] = Field(default=None)