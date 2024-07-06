def convert_bsf(data: str, bsf_markup: str, converter: str = 'beios') -> str:
    """
    Convert data file with NER markup in Brat Standoff Format to BEIOS or IOB format.

    :param converter: iob or beios converter to use for document
    :param data: tokenized data to be converted. Each token separated with a space
    :param bsf_markup: Brat Standoff Format markup
    :return: data in BEIOS or IOB format https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging)
    """

    def join_simple_chunk(chunk: str) -> list:
        if len(chunk.strip()) == 0:
            return []
        # keep the newlines, but discard the non-newline whitespace
        tokens = re.split(r'(\n)|\s', chunk.strip())
        # the re will return None for splits which were not caught in a group
        tokens = [x for x in tokens if x is not None]
        return [token + ' O' if len(token.strip()) > 0 else token for token in tokens]

    converters = {'beios': format_token_as_beios, 'iob': format_token_as_iob}
    res = []
    markup = parse_bsf(bsf_markup)

    prev_idx = 0
    m_ln: BsfInfo
    for m_ln in markup:
        res += join_simple_chunk(data[prev_idx:m_ln.start_idx])

        convert_f = converters[converter]
        res.extend(convert_f(m_ln.token, m_ln.tag))
        prev_idx = m_ln.end_idx

    if prev_idx < len(data) - 1:
        res += join_simple_chunk(data[prev_idx:])

    return '\n'.join(res)

def parse_bsf(bsf_data: str) -> list:
    """
    Convert textual bsf representation to a list of named entities.

    :param bsf_data: data in the format 'T9	PERS 778 783    токен'
    :return: list of named tuples for each line of the data representing a single named entity token
    """
    if len(bsf_data.strip()) == 0:
        return []

    ln_ptrn = re.compile(r'(T\d+)\s(\w+)\s(\d+)\s(\d+)\s(.+?)(?=T\d+\s\w+\s\d+\s\d+|$)', flags=re.DOTALL)
    result = []
    for m in ln_ptrn.finditer(bsf_data.strip()):
        bsf = BsfInfo(m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), m.group(5).strip())
        result.append(bsf)
    return result

