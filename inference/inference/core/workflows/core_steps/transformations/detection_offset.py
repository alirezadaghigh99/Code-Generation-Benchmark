class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
        }
    )
    type: Literal["DetectionOffset"]
    predictions: StepOutputSelector(
        kind=[
            BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
            BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
            BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Reference to detection-like predictions",
        examples=["$steps.object_detection_model.predictions"],
    )
    offset_width: Union[PositiveInt, WorkflowParameterSelector(kind=[INTEGER_KIND])] = (
        Field(
            description="Offset for boxes width",
            examples=[10, "$inputs.offset_x"],
            validation_alias=AliasChoices("offset_width", "offset_x"),
        )
    )
    offset_height: Union[
        PositiveInt, WorkflowParameterSelector(kind=[INTEGER_KIND])
    ] = Field(
        description="Offset for boxes height",
        examples=[10, "$inputs.offset_y"],
        validation_alias=AliasChoices("offset_height", "offset_y"),
    )

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions",
                kind=[
                    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
                    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
                    BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
                ],
            ),
        ]

