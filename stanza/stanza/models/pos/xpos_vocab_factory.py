def xpos_vocab_factory(data, shorthand):
    if shorthand not in XPOS_DESCRIPTIONS:
        logger.warning("%s is not a known dataset.  Examining the data to choose which xpos vocab to use", shorthand)
    desc = choose_simplest_factory(data, shorthand)
    if shorthand in XPOS_DESCRIPTIONS:
        if XPOS_DESCRIPTIONS[shorthand] != desc:
            # log instead of throw
            # otherwise, updating datasets would be unpleasant
            logger.error("XPOS tagset in %s has apparently changed!  Was %s, is now %s", shorthand, XPOS_DESCRIPTIONS[shorthand], desc)
    else:
        logger.warning("Chose %s for the xpos factory for %s", desc, shorthand)
    return build_xpos_vocab(desc, data, shorthand)

