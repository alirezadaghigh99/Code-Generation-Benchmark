def to_text(sentence):
    """
    Helper routine that converts a Sentence protobuf to a string from
    its tokens.
    """
    text = ""
    for i, tok in enumerate(sentence.token):
        if i != 0:
            text += tok.before
        text += tok.word
    return text

