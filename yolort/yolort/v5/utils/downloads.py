def attempt_download(file, repo="ultralytics/yolov5", hash_prefix=None):
    # Attempt file download if does not exist
    file = Path(str(file).strip().replace("'", ""))

    if not file.exists():
        # URL specified
        name = Path(urllib.parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        if str(file).startswith(("http:/", "https:/")):  # download
            url = str(file).replace(":/", "://")  # Pathlib turns :// -> :/
            name = name.split("?")[0]  # parse authentication https://url.com/file.txt?auth...
            safe_download(file=name, url=url, min_bytes=1e5, hash_prefix=hash_prefix)
            return name

        # GitHub assets
        file.parent.mkdir(parents=True, exist_ok=True)
        try:
            # github api
            response = requests.get(f"https://api.github.com/repos/{repo}/releases/latest").json()
            assets = [x["name"] for x in response["assets"]]
            tag = response["tag_name"]  # i.e. 'v1.0'
        except Exception as e:  # fallback plan
            print(f"Wrong when calling GitHub API: {e}")
            assets = [
                "yolov5n.pt",
                "yolov5s.pt",
                "yolov5m.pt",
                "yolov5l.pt",
                "yolov5x.pt",
                "yolov5n6.pt",
                "yolov5s6.pt",
                "yolov5m6.pt",
                "yolov5l6.pt",
                "yolov5x6.pt",
            ]
            try:
                tag = (
                    subprocess.check_output("git tag", shell=True, stderr=subprocess.STDOUT)
                    .decode()
                    .split()[-1]
                )
            except Exception as e:
                print(f"Wrong when getting GitHub tag: {e}")
                tag = "v6.0"  # current release

        if name in assets:
            safe_download(
                file,
                url=f"https://github.com/{repo}/releases/download/{tag}/{name}",
                min_bytes=1e5,
                error_msg=f"{file} missing, try downloading from https://github.com/{repo}/releases/",
            )

    return str(file)