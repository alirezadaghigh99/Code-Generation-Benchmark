def format_confusion(confusion, labels=None, hide_zeroes=False, hide_blank=False, transpose=False):
    """
    pretty print for confusion matrixes
    adapted from https://gist.github.com/zachguo/10296432

    The matrix should look like this:
      confusion[gold][pred]
    """
    def sort_labels(labels):
        """
        Sorts the labels in the list, respecting BIES if all labels are BIES, putting O at the front
        """
        labels = set(labels)
        if 'O' in labels:
            had_O = True
            labels.remove('O')
        else:
            had_O = False

        if not all(len(x) > 2 and x[0] in ('B', 'I', 'E', 'S') and x[1] in ('-', '_') for x in labels):
            labels = sorted(labels)
        else:
            # sort first by the body of the lable, then by BEIS
            labels = sorted(labels, key=lambda x: (x[2:], x[0]))
        if had_O:
            labels = ['O'] + labels
        return labels

    if transpose:
        new_confusion = defaultdict(lambda: defaultdict(int))
        for label1 in confusion.keys():
            for label2 in confusion[label1].keys():
                new_confusion[label2][label1] = confusion[label1][label2]
        confusion = new_confusion

    if labels is None:
        gold_labels = set(confusion.keys())
        if hide_blank:
            gold_labels = set(x for x in gold_labels if any(confusion[x][key] != 0 for key in confusion[x].keys()))

        pred_labels = set()
        for key in confusion.keys():
            if hide_blank:
                new_pred_labels = set(x for x in confusion[key].keys() if confusion[key][x] != 0)
            else:
                new_pred_labels = confusion[key].keys()
            pred_labels = pred_labels.union(new_pred_labels)

        if not hide_blank:
            gold_labels = gold_labels.union(pred_labels)
            pred_labels = gold_labels

        gold_labels = sort_labels(gold_labels)
        pred_labels = sort_labels(pred_labels)
    else:
        gold_labels = labels
        pred_labels = labels

    columnwidth = max([len(x) for x in pred_labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    # If the numbers are all ints, no need to include the .0 at the end of each entry
    all_ints = True
    for i, label1 in enumerate(gold_labels):
        for j, label2 in enumerate(pred_labels):
            if not isinstance(confusion.get(label1, {}).get(label2, 0), int):
                all_ints = False
                break
        if not all_ints:
            break

    if all_ints:
        format_cell = lambda confusion_cell: "%{0}d".format(columnwidth) % confusion_cell
    else:
        format_cell = lambda confusion_cell: "%{0}.1f".format(columnwidth) % confusion_cell

    # make sure the columnwidth can handle long numbers
    for i, label1 in enumerate(gold_labels):
        for j, label2 in enumerate(pred_labels):
            cell = confusion.get(label1, {}).get(label2, 0)
            columnwidth = max(columnwidth, len(format_cell(cell)))

    # if this is an NER confusion matrix (well, if it has - in the labels)
    # try to drop a bunch of labels to make the matrix easier to display
    if columnwidth * len(pred_labels) > 150:
        confusion, gold_labels, pred_labels = condense_ner_labels(confusion, gold_labels, pred_labels)

    # Print header
    if transpose:
        corner_label = "p\\t"
    else:
        corner_label = "t\\p"
    fst_empty_cell = (columnwidth-3)//2 * " " + corner_label + (columnwidth-3)//2 * " "
    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    header = "    " + fst_empty_cell + " "
    for label in pred_labels:
        header = header + "%{0}s ".format(columnwidth) % label
    text = [header.rstrip()]

    # Print rows
    for i, label1 in enumerate(gold_labels):
        row = "    %{0}s ".format(columnwidth) % label1
        for j, label2 in enumerate(pred_labels):
            confusion_cell = confusion.get(label1, {}).get(label2, 0)
            cell = format_cell(confusion_cell)
            if hide_zeroes:
                cell = cell if confusion_cell else empty_cell
            row = row + cell + " "
        text.append(row.rstrip())
    return "\n".join(text)

