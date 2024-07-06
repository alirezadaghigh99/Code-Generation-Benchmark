def get_wordvec_file(wordvec_dir, shorthand, wordvec_type=None):
    """ Lookup the name of the word vectors file, given a directory and the language shorthand.
    """
    lcode, tcode = shorthand.split('_', 1)
    lang = lcode2lang[lcode]
    # locate language folder
    word2vec_dir = os.path.join(wordvec_dir, 'word2vec', lang)
    fasttext_dir = os.path.join(wordvec_dir, 'fasttext', lang)
    lang_dir = None
    if wordvec_type is not None:
        lang_dir = os.path.join(wordvec_dir, wordvec_type, lang)
        if not os.path.exists(lang_dir):
            raise FileNotFoundError("Word vector type {} was specified, but directory {} does not exist".format(wordvec_type, lang_dir))
    elif os.path.exists(word2vec_dir): # first try word2vec
        lang_dir = word2vec_dir
    elif os.path.exists(fasttext_dir): # otherwise try fasttext
        lang_dir = fasttext_dir
    else:
        raise FileNotFoundError("Cannot locate word vector directory for language: {}  Looked in {} and {}".format(lang, word2vec_dir, fasttext_dir))
    # look for wordvec filename in {lang_dir}
    filename = os.path.join(lang_dir, '{}.vectors'.format(lcode))
    if os.path.exists(filename + ".xz"):
        filename = filename + ".xz"
    elif os.path.exists(filename + ".txt"):
        filename = filename + ".txt"
    return filename

