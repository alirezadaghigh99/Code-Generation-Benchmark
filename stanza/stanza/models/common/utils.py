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

def set_random_seed(seed):
    """
    Set a random seed on all of the things which might need it.
    torch, np, python random, and torch.cuda
    """
    if seed is None:
        seed = random.randint(0, 1000000000)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # some of these calls are probably redundant
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed

