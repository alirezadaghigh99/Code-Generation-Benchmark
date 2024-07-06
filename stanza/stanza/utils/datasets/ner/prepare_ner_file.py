def process_dataset(input_filename, output_filename):
    sentences = load_conll03(input_filename)
    print("{} examples loaded from {}".format(len(sentences), input_filename))
    
    document = []
    for (words, tags) in sentences:
        sent = []
        for w, t in zip(words, tags):
            sent += [{'text': w, 'ner': t}]
        document += [sent]

    with open(output_filename, 'w', encoding="utf-8") as outfile:
        json.dump(document, outfile, indent=1)
    print("Generated json file {}".format(output_filename))

