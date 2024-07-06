def maybe_add_fake_dependencies(lines):
    """
    Possibly add fake dependencies in columns 6 and 7 (counting from 0)

    The conllu scripts need the dependencies column filled out, so in
    the case of models we build without dependency data, we need to
    add those fake dependencies in order to use the eval script etc

    lines: a list of strings with 10 tab separated columns
      comments are allowed (they will be skipped)

    returns: the same strings, but with fake dependencies added
      if columns 6 and 7 were empty
    """
    new_lines = []
    root_idx = None
    first_idx = None
    for line_idx, line in enumerate(lines):
        if line.startswith("#"):
            new_lines.append(line)
            continue

        pieces = line.split("\t")
        if MWT_OR_COPY_RE.match(pieces[0]):
            new_lines.append(line)
            continue

        token_idx = int(pieces[0])
        if pieces[6] != '_':
            if pieces[6] == '0':
                root_idx = token_idx
            new_lines.append(line)
        elif token_idx == 1:
            # note that the comments might make this not the first line
            # we keep track of this separately so we can either make this the root,
            # or set this to be the root later
            first_idx = line_idx
            new_lines.append(pieces)
        else:
            pieces[6] = "1"
            pieces[7] = "dep"
            new_lines.append("\t".join(pieces))
    if first_idx is not None:
        if root_idx is None:
            new_lines[first_idx][6] = "0"
            new_lines[first_idx][7] = "root"
        else:
            new_lines[first_idx][6] = str(root_idx)
            new_lines[first_idx][7] = "dep"
        new_lines[first_idx] = "\t".join(new_lines[first_idx])
    return new_lines

