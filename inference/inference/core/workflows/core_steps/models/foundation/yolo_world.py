class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": "Run a zero-shot object detection model.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
        }
    )
    type: Literal["YoloWorldModel", "YoloWorld"]
    name: str = Field(description="Unique name of step in workflows")
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference an image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("images", "image"),
    )
    class_names: Union[
        WorkflowParameterSelector(kind=[LIST_OF_VALUES_KIND]), List[str]
    ] = Field(
        description="One or more classes that you want YOLO-World to detect. The model accepts any string as an input, though does best with short descriptions of common objects.",
        examples=[["person", "car", "license plate"], "$inputs.class_names"],
    )
    version: Union[
        Literal[
            "v2-s",
            "v2-m",
            "v2-l",
            "v2-x",
            "s",
            "m",
            "l",
            "x",
        ],
        WorkflowParameterSelector(kind=[STRING_KIND]),
    ] = Field(
        default="v2-s",
        description="Variant of YoloWorld model",
        examples=["v2-s", "$inputs.variant"],
    )
    confidence: Union[
        Optional[FloatZeroToOne],
        WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.005,
        description="Confidence threshold for detections",
        examples=[0.005, "$inputs.confidence"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="parent_id", kind=[BATCH_OF_PARENT_ID_KIND]),
            OutputDefinition(
                name="predictions", kind=[BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND]
            ),
            OutputDefinition(name="image", kind=[BATCH_OF_IMAGE_METADATA_KIND]),
            OutputDefinition(
                name="prediction_type", kind=[BATCH_OF_PREDICTION_TYPE_KIND]
            ),
        ]