def save_each_file_name(args):
    model_file = model_file_name(args)
    pieces = os.path.splitext(model_file)
    return pieces[0] + "_%05d" + pieces[1]

