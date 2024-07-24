class InContextLearningGenerationExactMatchAccuracy(InContextLearningMetric):
    r"""Computes exact match for in-context learning generation tasks.

    ICL generation tasks consist of some number of prompted generation tasks with correct answers
    followed by a test task where the model must correctly produce one of a number of valid answers.

    For example, the model may be provided the context below and evaluated on its ability to correctly predict the continuation.

    Context: `Question: Who was president of the United States in 2012?\nAnswer: Barack Obama\nQuestion: Is water wet?\nAnswer: `
    Answers: [`yes`]

    The model will be expected to correctly produce one of the answers, following some optional normalization.

    Adds metric state variables:
        correct (float): The number of instances where the prediction was a prefix for any of the answer aliases.
        total (float): The number of total instances that were predicted.

    Args:
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False):
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            'correct',
            default=torch.tensor(0.),
            dist_reduce_fx='sum',
        )
        self.add_state('total', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.metric_result_dict = {
            'cleaned_output': [],
            'original_label': [],
            'cleaned_label': [],
            'result': [],
        }

    def normalize_answer(self, answer: str):
        """Lower text and remove punctuation, articles and extra whitespace.

        Copied from https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py
        """

        def remove_articles(text: str) -> str:
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text: str) -> str:
            return ' '.join(text.split())

        def handle_punc(text: str) -> str:
            exclude = set(
                string.punctuation + ''.join([u'‘', u'’', u'´', u'`']),
            )
            return ''.join(ch if ch not in exclude else ' ' for ch in text)

        def lower(text: str) -> str:
            return text.lower()

        def replace_underscore(text: str) -> str:
            return text.replace('_', ' ')

        return white_space_fix(
            remove_articles(handle_punc(lower(replace_underscore(answer)))),
        ).strip()

    def update(
        self,
        batch: Dict[str, Any],
        outputs: List[str],
        labels: List[List[str]],
    ):
        cot_delimiter = batch.get('cot_delimiter', '')
        do_normalization = batch.get('do_normalization', True)
        stopping_criteria = batch.get('stopping_criteria', None)
        metric_result_dict = copy.deepcopy(self.metric_result_dict)
        for sample_output, sample_labels in zip(outputs, labels):
            final_answer = sample_output

            if stopping_criteria is not None and len(stopping_criteria) > 0:
                final_answer = re.split(
                    '|'.join(stopping_criteria),
                    final_answer,
                )[0]

            if cot_delimiter is not None and len(cot_delimiter) > 0:
                final_answer = final_answer.split(cot_delimiter)[-1]

            if do_normalization:
                cleaned_final_answer = self.normalize_answer(final_answer)
                cleaned_sample_labels = {
                    self.normalize_answer(label) for label in sample_labels
                }
            else:
                # even if normalization is off, we should still strip leading/trailing whitespaces
                cleaned_final_answer = final_answer.strip()
                cleaned_sample_labels = {
                    sample_label.strip() for sample_label in sample_labels
                }
            metric_result_dict['original_label'].append(sample_labels)
            metric_result_dict['cleaned_output'].append(cleaned_final_answer)
            metric_result_dict['cleaned_label'].append(cleaned_sample_labels)

            if any(
                cleaned_final_answer.startswith(label)
                for label in cleaned_sample_labels
            ):
                self.correct += torch.tensor(1.0)
                metric_result_dict['result'].append(1)
            else:
                metric_result_dict['result'].append(0)

            self.total += torch.tensor(1.0)

        return metric_result_dict

    def compute(self):
        assert isinstance(self.correct, Tensor)
        assert isinstance(self.total, Tensor)
        return self.correct / self.total

class InContextLearningMultipleChoiceAccuracy(InContextLearningMetric):
    r"""Computes accuracy for In-context learning multiple choice tasks.

    ICL MC tasks consists of a series of questions with some number of possible choices (only one of which can be correct).
    At inference time each possible choice is given to the model as a separate input and the one for which the model assigns
    the lowest perplexity to the choice is considered the model's choice. The model is correct if it "chooses" the right answer.

    Context: `The dog is->fuzzy\nthe water is->hot\nthe tree is->`
    Continuation: `green`

    Adds metric state variables:
        correct (float): The number of instances where the prediction masked the target.
        total (float): The number of total instances that were predicted.

    Args:
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False):
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            'correct',
            default=torch.tensor(0.0),
            dist_reduce_fx='sum',
        )
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.metric_result_dict = {
            'context': [],
            'correct_choice': [],
            'correct_choice_idx': [],
            'selected_choice': [],
            'selected_choice_idx': [],
            'all_choices': [],
            'result': [],
        }

    def update(self, batch: dict, outputs: torch.Tensor, labels: torch.Tensor):

        perplexities = []
        for batch_idx, cont_idx in enumerate(batch['continuation_indices']):
            # continuation indices refer to indices in the original input's token space
            cont_tok_logits = outputs[batch_idx].index_select(
                dim=0,
                index=cont_idx - 1,
            )
            # labels have been shifted left by one index, so the cont_idx needs to be shifted as well.
            cont_tok_targ = labels[batch_idx].index_select(
                dim=0,
                index=cont_idx - 1,
            )
            cross_entropy = F.cross_entropy(cont_tok_logits, cont_tok_targ)
            perplexity = torch.exp(cross_entropy)
            perplexities.append(perplexity)

        metric_result_dict = copy.deepcopy(self.metric_result_dict)
        for (start, end), gold_idx in zip(
            batch['choice_groupings'],
            batch['gold_indices'],
        ):
            subset = perplexities[start:end]
            idx_min = subset.index(min(subset))

            if idx_min == gold_idx:
                self.correct += torch.tensor(1.0)
                metric_result_dict['result'].append(1)
            else:
                metric_result_dict['result'].append(0)

            question = batch['input_ids'][
                start][:batch['continuation_indices'][start][0]]

            correct_choice = batch['input_ids'][start:end][gold_idx][
                batch['continuation_indices'][start:end][gold_idx][0]:
                batch['continuation_indices'][start:end][gold_idx][-1] + 1]
            selected_choice = batch['input_ids'][start:end][idx_min][
                batch['continuation_indices'][start:end][idx_min][0]:
                batch['continuation_indices'][start:end][idx_min][-1] + 1]
            metric_result_dict['context'].append(question)
            metric_result_dict['correct_choice'].append(correct_choice)
            metric_result_dict['correct_choice_idx'].append(gold_idx)
            metric_result_dict['selected_choice'].append(selected_choice)
            metric_result_dict['selected_choice_idx'].append(idx_min)
            all_choices = batch['input_ids'][start:end]
            # Unpads the choices. Necessary in case different choices have different token lengths.
            if 'attention_mask' in batch:
                all_choices_list = [
                    choice[batch['attention_mask'][i]]
                    for i, choice in enumerate(all_choices)
                ]
                metric_result_dict['all_choices'].append(all_choices_list)

            self.total += torch.tensor(1.0)

        # Don't return all_choices if we didn't fill it up (i.e. didn't use causal lms)
        if metric_result_dict['all_choices'] == []:
            metric_result_dict.pop('all_choices')

        return metric_result_dict

    def compute(self):
        assert isinstance(self.correct, Tensor)
        assert isinstance(self.total, Tensor)
        return self.correct.float() / self.total

class InContextLearningLMAccuracy(InContextLearningMetric):
    r"""Computes accuracy for In-context learning language modeling tasks.

    ICL LM tasks consist of some number of example language modeling tasks (referred to as the 'context'), followed by a test task where the model must correctly predict all the tokens
    following tokens in some passage (referred to as the 'continuation').

    For example, the model may be provided the context below and evaluated on its ability to correctly predict the continuation. Note: it doesn't matter
    whether the model correctly predicts the context tokens.

    Context: `The dog is->fuzzy\nthe water is->hot\nthe tree is->`
    Continuation: `green`

    Adds metric state variables:
        correct (float): The number of instances where the prediction masked the target.
        total (float): The number of total instances that were predicted.

    Args:
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False):
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            'correct',
            default=torch.tensor(0.),
            dist_reduce_fx='sum',
        )
        self.add_state('total', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.metric_result_dict = {
            'context': [],
            'label': [],
            'output': [],
            'result': [],
        }

    def update(self, batch: dict, outputs: torch.Tensor, labels: torch.Tensor):

        metric_result_dict = copy.deepcopy(self.metric_result_dict)
        for batch_idx, cont_idx in enumerate(batch['continuation_indices']):
            cont_tok_pred = outputs[batch_idx].index_select(
                dim=0,
                index=cont_idx - 1,
            ).argmax(dim=-1)
            cont_tok_targ = labels[batch_idx].index_select(
                dim=0,
                index=cont_idx - 1,
            )

            metric_result_dict['context'].append(
                batch['input_ids'][batch_idx][:cont_idx[0]],
            )
            metric_result_dict['label'].append(cont_tok_targ)
            metric_result_dict['output'].append(cont_tok_pred)

            correct = (cont_tok_pred == cont_tok_targ).all().int()
            self.correct += correct
            metric_result_dict['result'].append(int(correct))

            self.total += torch.tensor(1.0)

        return metric_result_dict

    def compute(self):
        assert isinstance(self.correct, Tensor)
        assert isinstance(self.total, Tensor)
        return self.correct / self.total

