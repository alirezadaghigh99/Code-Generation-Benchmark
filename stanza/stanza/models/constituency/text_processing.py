def parse_tokenized_sentences(args, model, retag_pipeline, sentences):
    """
    Parse the given sentences, return a list of ParseResult objects
    """
    tags = retag_tags(sentences, retag_pipeline, model.uses_xpos())
    words = [[(word, tag) for word, tag in zip(s_words, s_tags)] for s_words, s_tags in zip(sentences, tags)]
    logger.info("Retagging finished.  Parsing tagged text")

    assert len(words) == len(sentences)
    treebank = model.parse_sentences_no_grad(iter(tqdm(words)), model.build_batch_from_tagged_words, args['eval_batch_size'], model.predict, keep_scores=False)
    return treebank

