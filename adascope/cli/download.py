from pathlib import Path

DEFAULT_URL = "https://www.youtube.com/watch?v=0Ol8MA9e8nM"


def configure_parser(parser):
    parser.add_argument("url_arg", nargs="?", help="YouTube URL")
    parser.add_argument("--url", dest="url_opt")
    parser.add_argument("--output-dir", help="Compatibility output directory")
    parser.add_argument("--out", default="data/raw/cluster_video.mp4")
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--ffmpeg-location")
    parser.add_argument("--force", action="store_true")


def run(args):
    url = args.url_opt or args.url_arg
    url = url or DEFAULT_URL
    try:
        import yt_dlp
    except ImportError:
        raise SystemExit("yt-dlp is required: pip install -e .[download]")
    out = Path(args.out)
    if args.output_dir:
        out = Path(args.output_dir) / "%(id)s.%(ext)s"
    elif out.exists() and not args.force:
        print(f"Already exists: {out} (use --force to overwrite)")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    options = {
        "format": f"bestvideo[height<={args.height}]+bestaudio/best[height<={args.height}]",
        "outtmpl": str(out), "merge_output_format": "mp4", "overwrites": args.force,
    }
    if args.ffmpeg_location:
        options["ffmpeg_location"] = args.ffmpeg_location
    with yt_dlp.YoutubeDL(options) as client:
        client.download([url])
    return 0
