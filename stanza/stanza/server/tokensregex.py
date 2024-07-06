def process_doc(doc, *patterns):
    request = TokensRegexRequest()
    for pattern in patterns:
        request.pattern.append(pattern)

    request_doc = request.doc
    request_doc.text = doc.text
    num_tokens = 0
    for sentence in doc.sentences:
        add_sentence(request_doc.sentence, sentence, num_tokens)
        num_tokens = num_tokens + sum(len(token.words) for token in sentence.tokens)

    return send_tokensregex_request(request)

