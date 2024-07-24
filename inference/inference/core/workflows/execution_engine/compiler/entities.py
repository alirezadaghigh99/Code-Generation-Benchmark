class OutputNode(ExecutionGraphNode):
    output_manifest: JsonField

    @property
    def dimensionality(self) -> int:
        return len(self.data_lineage)

    def is_batch_oriented(self) -> bool:
        return len(self.data_lineage) > 0

class StepNode(ExecutionGraphNode):
    step_manifest: WorkflowBlockManifest
    input_data: StepInputData = field(default_factory=dict)
    dimensionality_reference_property: Optional[str] = None
    child_execution_branches: Dict[str, str] = field(default_factory=dict)
    execution_branches_impacting_inputs: Set[str] = field(default_factory=set)
    batch_oriented_parameters: Set[str] = field(default_factory=set)
    step_execution_dimensionality: int = 0

    def controls_flow(self) -> bool:
        if self.child_execution_branches:
            return True
        return False

    @property
    def output_dimensionality(self) -> int:
        return len(self.data_lineage)

    def is_batch_oriented(self) -> bool:
        return len(self.batch_oriented_parameters) > 0

