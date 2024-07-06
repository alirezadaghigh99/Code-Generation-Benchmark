def compare_ignoring_whitespace(predicted, expected):
    predicted = re.sub('[ \t]+', ' ', predicted.strip())
    predicted = re.sub('\r\n', '\n', predicted)
    expected = re.sub('[ \t]+', ' ', expected.strip())
    expected = re.sub('\r\n', '\n', expected)
    assert predicted == expected

