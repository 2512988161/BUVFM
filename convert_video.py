#!/usr/bin/env python3
"""Convert video to H.264 (libx264) for browser compatibility."""
import argparse
import sys

import imageio.v3 as iio


def main():
    parser = argparse.ArgumentParser(description="Convert video to H.264")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", help="Output video path")
    parser.add_argument("--fps", type=int, default=None, help="Output FPS (default: keep source FPS)")
    args = parser.parse_args()

    try:
        frames = iio.imread(args.input, plugin="pyav")
        meta = iio.immeta(args.input, plugin="pyav")
        fps = args.fps if args.fps else int(meta.get("fps", 8))
        iio.imwrite(args.output, frames, fps=fps, codec="libx264")
        print(f"Converted {args.input} -> {args.output} ({len(frames)} frames, {fps} fps)")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
