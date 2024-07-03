class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "flow_control",
        }
    )
    type: Literal["Condition"]
    left: Union[
        float,
        int,
        bool,
        StepOutputSelector(
            kind=[
                FLOAT_KIND,
                INTEGER_KIND,
                BOOLEAN_KIND,
                STRING_KIND,
                LIST_OF_VALUES_KIND,
            ]
        ),
        WorkflowParameterSelector(
            kind=[
                FLOAT_KIND,
                INTEGER_KIND,
                BOOLEAN_KIND,
                STRING_KIND,
                LIST_OF_VALUES_KIND,
                WILDCARD_KIND,
            ]
        ),
        str,
        list,
        set,
    ] = Field(
        description="Left operand of expression `left operator right` to evaluate boolean value of condition statement",
        examples=["$steps.classification.top", 3, "foo"],
    )
    operator: Operator = Field(
        description="Operator in expression `left operator right` to evaluate boolean value of condition statement",
        examples=["equal", "in"],
    )
    right: Union[
        float,
        int,
        bool,
        StepOutputSelector(
            kind=[
                FLOAT_KIND,
                INTEGER_KIND,
                BOOLEAN_KIND,
                STRING_KIND,
                LIST_OF_VALUES_KIND,
            ]
        ),
        WorkflowParameterSelector(
            kind=[
                FLOAT_KIND,
                INTEGER_KIND,
                BOOLEAN_KIND,
                STRING_KIND,
                LIST_OF_VALUES_KIND,
                WILDCARD_KIND,
            ]
        ),
        str,
        list,
        set,
    ] = Field(
        description="Right operand of expression `left operator right` to evaluate boolean value of condition statement",
        examples=["$steps.classification.top", 3, "bar"],
    )
    step_if_true: StepSelector = Field(
        description="Reference to step which shall be executed if expression evaluates to true",
        examples=["$steps.on_true"],
    )
    step_if_false: StepSelector = Field(
        description="Reference to step which shall be executed if expression evaluates to false",
        examples=["$steps.on_false"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []