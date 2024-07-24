def try_parse_lmm_output_to_json(
    output: str, expected_output: Dict[str, str]
) -> Union[list, dict]:
    json_blocks_found = JSON_MARKDOWN_BLOCK_PATTERN.findall(output)
    if len(json_blocks_found) == 0:
        return try_parse_json(output, expected_output=expected_output)
    result = []
    for json_block in json_blocks_found:
        result.append(
            try_parse_json(content=json_block, expected_output=expected_output)
        )
    return result if len(result) > 1 else result[0]

def try_parse_json(content: str, expected_output: Dict[str, str]) -> dict:
    try:
        data = json.loads(content)
        return {key: data.get(key, NOT_DETECTED_VALUE) for key in expected_output}
    except Exception:
        return {key: NOT_DETECTED_VALUE for key in expected_output}

class LMMConfig(BaseModel):
    max_tokens: int = Field(default=450)
    gpt_image_detail: Literal["low", "high", "auto"] = Field(
        default="auto",
        description="To be used for GPT-4V only.",
    )
    gpt_model_version: str = Field(default="gpt-4o")

class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": "Run a large language model.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
        }
    )
    type: Literal["LMM"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = ImageInputField
    prompt: Union[WorkflowParameterSelector(kind=[STRING_KIND]), str] = Field(
        description="Holds unconstrained text prompt to LMM mode",
        examples=["my prompt", "$inputs.prompt"],
    )
    lmm_type: Union[
        WorkflowParameterSelector(kind=[STRING_KIND]), Literal["gpt_4v", "cog_vlm"]
    ] = Field(
        description="Type of LMM to be used", examples=["gpt_4v", "$inputs.lmm_type"]
    )
    lmm_config: LMMConfig = Field(
        default_factory=lambda: LMMConfig(), description="Configuration of LMM"
    )
    remote_api_key: Union[
        WorkflowParameterSelector(kind=[STRING_KIND]), Optional[str]
    ] = Field(
        default=None,
        description="Holds API key required to call LMM model - in current state of development, we require OpenAI key when `lmm_type=gpt_4v` and do not require additional API key for CogVLM calls.",
        examples=["xxx-xxx", "$inputs.api_key"],
    )
    json_output: Optional[Dict[str, str]] = Field(
        default=None,
        description="Holds dictionary that maps name of requested output field into its description",
        examples=[{"count": "number of cats in the picture"}, "$inputs.json_output"],
    )

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="parent_id", kind=[BATCH_OF_PARENT_ID_KIND]),
            OutputDefinition(name="root_parent_id", kind=[BATCH_OF_PARENT_ID_KIND]),
            OutputDefinition(name="image", kind=[BATCH_OF_IMAGE_METADATA_KIND]),
            OutputDefinition(name="structured_output", kind=[BATCH_OF_DICTIONARY_KIND]),
            OutputDefinition(name="raw_output", kind=[BATCH_OF_STRING_KIND]),
            OutputDefinition(name="*", kind=[WILDCARD_KIND]),
        ]

    def get_actual_outputs(self) -> List[OutputDefinition]:
        result = [
            OutputDefinition(name="parent_id", kind=[BATCH_OF_PARENT_ID_KIND]),
            OutputDefinition(name="root_parent_id", kind=[BATCH_OF_PARENT_ID_KIND]),
            OutputDefinition(name="image", kind=[BATCH_OF_IMAGE_METADATA_KIND]),
            OutputDefinition(name="structured_output", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="raw_output", kind=[STRING_KIND]),
        ]
        if self.json_output is None:
            return result
        for key in self.json_output.keys():
            result.append(OutputDefinition(name=key, kind=[WILDCARD_KIND]))
        return result

