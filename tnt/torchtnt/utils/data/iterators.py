class RandomizedBatchSampler(DataIterationStrategy):
    weights: Optional[Dict[str, float]] = None
    stopping_mechanism: StoppingMechanism = StoppingMechanism.ALL_DATASETS_EXHAUSTED
    enforce_same_loader_across_ranks: bool = False

class AllDatasetBatches(DataIterationStrategy):
    stopping_mechanism: StoppingMechanism = StoppingMechanism.ALL_DATASETS_EXHAUSTED

class RoundRobin(DataIterationStrategy):
    stopping_mechanism: StoppingMechanism = StoppingMechanism.ALL_DATASETS_EXHAUSTED
    iteration_order: Optional[List[str]] = None

class InOrder(DataIterationStrategy):
    iteration_order: Optional[List[str]] = None

class DataIterationStrategy:
    pass

