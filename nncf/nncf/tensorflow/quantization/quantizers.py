def _min_adj(bits, low, range_len, narrow_range):
        quants_count = 2**bits - (2 if narrow_range else 1)
        return range_len / quants_count * tf.round(quants_count * low / range_len)

