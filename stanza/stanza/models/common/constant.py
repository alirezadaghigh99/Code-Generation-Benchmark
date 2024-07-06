def lang_to_langcode(lang):
    if lang in lang2lcode:
        lcode = lang2lcode[lang]
    elif lang.lower() in langlower2lcode:
        lcode = langlower2lcode[lang.lower()]
    elif lang in lcode2lang:
        lcode = lang
    elif lang.lower() in lcode2lang:
        lcode = lang.lower()
    else:
        raise ValueError("Unable to find language code for %s" % lang)
    return lcode

