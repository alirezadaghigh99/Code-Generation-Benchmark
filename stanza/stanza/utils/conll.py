def conll2doc(input_file=None, input_str=None, ignore_gapping=True, zip_file=None):
        doc_dict, doc_comments, doc_empty = CoNLL.conll2dict(input_file, input_str, ignore_gapping, zip_file=zip_file)
        return Document(doc_dict, text=None, comments=doc_comments, empty_sentences=doc_empty)

