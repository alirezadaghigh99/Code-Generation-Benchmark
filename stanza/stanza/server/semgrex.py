def process_doc(doc, *semgrex_patterns, enhanced=False):
    """
    Returns the result of processing the given semgrex expression on the stanza doc.

    Currently the return is a SemgrexResponse from CoreNLP.proto
    """
    request = build_request(doc, semgrex_patterns, enhanced=enhanced)

    return send_semgrex_request(request)

