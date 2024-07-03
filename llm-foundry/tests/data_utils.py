def make_tiny_ft_dataset(
    path: str,
    size: int = 4,
    add_bad_data_dropped: bool = False,
    add_invalid_prompt_type: bool = False,
    add_invalid_response_type: bool = False,
    add_unknown_example_type: bool = False,
    add_just_bos_eos_pad: bool = False,
    add_too_many_example_keys: bool = False,
    pad_token: Optional[str] = None,
    start_token: Optional[str] = None,
    end_token: Optional[str] = None,
):
    if Path(path).suffix != '.jsonl':
        raise ValueError(f'Path {path} must be a jsonl file.')
    good_sample = {'prompt': 'hello', 'response': 'goodbye'}
    samples = [good_sample] * size
    if add_bad_data_dropped:
        if pad_token is None:
            raise ValueError(
                'pad_token, start_token, and end_token must be specified if add_bad_data is True',
            )
        # empty prompt
        samples.append({'prompt': '', 'response': 'goodbye'})
        # empty response
        samples.append({'prompt': 'hello', 'response': ''})

    if add_invalid_prompt_type:
        # prompt just None
        samples.append({
            'prompt': None,
            'response': 'goodbye',
        })  # type: ignore (intentional test)

    if add_invalid_response_type:
        # response just None
        samples.append({
            'prompt': 'hello',
            'response': None,
        })  # type: ignore (intentional test)

    if add_too_many_example_keys:
        # too many keys
        samples.append({
            'prompt': 'hello',
            'response': 'goodbye',
            'completion': 'bar',
        })

    if add_just_bos_eos_pad:
        if pad_token is None or start_token is None or end_token is None:
            raise ValueError(
                'pad_token, start_token, and end_token must be specified if add_just_bos_eos is True',
            )
        # prompt just start
        samples.append({'prompt': start_token, 'response': 'goodbye'})
        # response just start
        samples.append({'prompt': 'hello', 'response': start_token})
        # prompt just end
        samples.append({'prompt': end_token, 'response': 'goodbye'})
        # response just end
        samples.append({'prompt': 'hello', 'response': end_token})
        # prompt just pad
        samples.append({'prompt': pad_token, 'response': 'goodbye'})
    if add_unknown_example_type:
        # unknown example type
        samples = [{'foo': 'yee', 'bar': 'haw'}]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as _f:
        for sample in samples:
            _f.write(json.dumps(sample))
            _f.write('\n')