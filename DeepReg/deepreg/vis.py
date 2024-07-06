def main(args=None):
    """
    CLI for deepreg_vis tool.

    Requires ffmpeg wirter to write gif files.

    :param args:
    """
    parser = argparse.ArgumentParser(
        description="deepreg_vis", formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--mode",
        "-m",
        help="Mode of visualisation \n"
        "0 for animtion over image slices, \n"
        "1 for warp animation, \n"
        "2 for tile plot",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--image-paths",
        "-i",
        help="File path for image file "
        "(can specify multiple paths using a comma separated string)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save-path",
        "-s",
        help="Path to directory where resulting visualisation is saved",
        default="",
    )

    parser.add_argument(
        "--interval",
        help="Interval between frames of animation (in miliseconds)\n"
        "Applicable only if --mode 0 or --mode 1 or --mode 3",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--ddf-path",
        help="Path to ddf used for warping images\n"
        "Applicable only and required if --mode 1",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--num-interval",
        help="Number of intervals to use for warping\n" "Applicable only if --mode 1",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--slice-inds",
        help="Comma separated string of indexes of slices"
        " to be used for the visualisation\n"
        "Applicable only if --mode 1 or --mode 2",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--fname",
        help="File name (with extension like .png, .jpeg, .gif, ...)"
        " to save visualisation to\n"
        "Applicable only if --mode 2 or --mode 3",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--col-titles",
        help="Comma separated string of column titles to use "
        "(inferred from file names if not provided)\n"
        "Applicable only if --mode 2",
        default=None,
    )
    parser.add_argument(
        "--size",
        help="Comma separated string of number of columns and rows (e.g. '2,2')\n"
        "Applicable only if --mode 3",
        default="2,2",
    )

    # init arguments
    args = parser.parse_args(args)

    if args.slice_inds is not None:
        args.slice_inds = string_to_list(args.slice_inds)
        args.slice_inds = [int(elem) for elem in args.slice_inds]

    if args.mode == 0:
        gif_slices(
            img_paths=args.image_paths, save_path=args.save_path, interval=args.interval
        )
    elif args.mode == 1:
        if args.ddf_path is None:
            raise Exception("--ddf-path is required when using --mode 1")
        gif_warp(
            img_paths=args.image_paths,
            ddf_path=args.ddf_path,
            slice_inds=args.slice_inds,
            num_interval=args.num_interval,
            interval=args.interval,
            save_path=args.save_path,
        )
    elif args.mode == 2:
        tile_slices(
            img_paths=args.image_paths,
            save_path=args.save_path,
            fname=args.fname,
            slice_inds=args.slice_inds,
            col_titles=args.col_titles,
        )
    elif args.mode == 3:
        size = tuple([int(elem) for elem in string_to_list(args.size)])
        gif_tile_slices(
            img_paths=args.image_paths,
            save_path=args.save_path,
            fname=args.fname,
            interval=args.interval,
            size=size,
        )

