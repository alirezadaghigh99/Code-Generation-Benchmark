class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Instance Segmentation Model",
            "short_description": "Predict the shape and size of objects.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
        },
        protected_namespaces=(),
    )
    type: Literal["RoboflowInstanceSegmentationModel", "InstanceSegmentationModel"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = ImageInputField
    model_id: Union[WorkflowParameterSelector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = (
        RoboflowModelField
    )
    class_agnostic_nms: Union[bool, WorkflowParameterSelector(kind=[BOOLEAN_KIND])] = (
        Field(
            default=False,
            description="Value to decide if NMS is to be used in class-agnostic mode.",
            examples=[True, "$inputs.class_agnostic_nms"],
        )
    )
    class_filter: Union[
        Optional[List[str]], WorkflowParameterSelector(kind=[LIST_OF_VALUES_KIND])
    ] = Field(
        default=None,
        description="List of classes to retrieve from predictions (to define subset of those which was used while model training)",
        examples=[["a", "b", "c"], "$inputs.class_filter"],
    )
    confidence: Union[
        FloatZeroToOne,
        WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.4,
        description="Confidence threshold for predictions",
        examples=[0.3, "$inputs.confidence_threshold"],
    )
    iou_threshold: Union[
        FloatZeroToOne,
        WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.3,
        description="Parameter of NMS, to decide on minimum box intersection over union to merge boxes",
        examples=[0.4, "$inputs.iou_threshold"],
    )
    max_detections: Union[
        PositiveInt, WorkflowParameterSelector(kind=[INTEGER_KIND])
    ] = Field(
        default=300,
        description="Maximum number of detections to return",
        examples=[300, "$inputs.max_detections"],
    )
    max_candidates: Union[
        PositiveInt, WorkflowParameterSelector(kind=[INTEGER_KIND])
    ] = Field(
        default=3000,
        description="Maximum number of candidates as NMS input to be taken into account.",
        examples=[3000, "$inputs.max_candidates"],
    )
    mask_decode_mode: Union[
        Literal["accurate", "tradeoff", "fast"],
        WorkflowParameterSelector(kind=[STRING_KIND]),
    ] = Field(
        default="accurate",
        description="Parameter of mask decoding in prediction post-processing.",
        examples=["accurate", "$inputs.mask_decode_mode"],
    )
    tradeoff_factor: Union[
        FloatZeroToOne,
        WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.0,
        description="Post-processing parameter to dictate tradeoff between fast and accurate",
        examples=[0.3, "$inputs.tradeoff_factor"],
    )
    disable_active_learning: Union[
        bool, WorkflowParameterSelector(kind=[BOOLEAN_KIND])
    ] = Field(
        default=True,
        description="Parameter to decide if Active Learning data sampling is disabled for the model",
        examples=[True, "$inputs.disable_active_learning"],
    )
    active_learning_target_dataset: Union[
        WorkflowParameterSelector(kind=[ROBOFLOW_PROJECT_KIND]), Optional[str]
    ] = Field(
        default=None,
        description="Target dataset for Active Learning data sampling - see Roboflow Active Learning "
        "docs for more information",
        examples=["my_project", "$inputs.al_target_project"],
    )

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions",
                kind=[BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND],
            ),
        ]

