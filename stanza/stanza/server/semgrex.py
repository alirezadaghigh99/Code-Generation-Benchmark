def process_doc(doc, *semgrex_patterns, enhanced=False):
    """
    Returns the result of processing the given semgrex expression on the stanza doc.

    Currently the return is a SemgrexResponse from CoreNLP.proto
    """
    request = build_request(doc, semgrex_patterns, enhanced=enhanced)

    return send_semgrex_request(request)

class Semgrex(JavaProtobufContext):
    """
    Semgrex context window

    This is a context window which keeps a process open.  Should allow
    for multiple requests without launching new java processes each time.
    """
    def __init__(self, classpath=None):
        super(Semgrex, self).__init__(classpath, SemgrexResponse, SEMGREX_JAVA)

    def process(self, doc, *semgrex_patterns):
        """
        Apply each of the semgrex patterns to each of the dependency trees in doc
        """
        request = build_request(doc, semgrex_patterns)
        return self.process_request(request)

