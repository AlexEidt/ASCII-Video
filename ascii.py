#!/usr/bin/env python3
"""
Alex Eidt

Converts videos/images into ASCII video/images in various formats.
"""

import os
import argparse
import imageio
import imageio_ffmpeg
import numpy as np
from tqdm import tqdm as ProgressBar
from PIL import Image, ImageFont, ImageDraw


def get_font_bitmaps(fontsize, boldness, reverse, background, chars, font):
    """
    Returns a list of font bitmaps.

    Parameters
        fontsize    - Font size to use for ASCII characters
        boldness    - Stroke size to use when drawing ASCII characters
        reverse     - Reverse the ordering of the ASCII characters
        background  - Background color
        chars       - ASCII characters to use in media
        font        - Font to use

    Returns
        List of font bitmaps corresponding to the characters in "chars".
    """
    bitmaps = {}
    min_width = min_height = float("inf")
    font_ttf = ImageFont.truetype(font, size=fontsize)

    for char in chars:
        if char in bitmaps:
            continue
        w, h = font_ttf.getsize(char)
        min_width, min_height = min(min_width, w), min(min_height, h)
        # Draw font character as a w x h image.
        image = Image.new("RGB", (w, h), (background,) * 3)
        draw = ImageDraw.Draw(image)
        draw.text(
            (0, -(fontsize // 6)),
            char,
            fill=(255 - background,) * 3,
            font=font_ttf,
            stroke_width=boldness,
        )
        bitmap = np.array(image)
        if background == 255:
            np.subtract(255, bitmap, out=bitmap)
        bitmaps[char] = bitmap.astype(np.uint8)

    # Crop the font bitmaps to all have the same dimensions based on the
    # minimum font width and height of all font bitmaps.
    fonts = [bitmaps[char][: int(min_height), : int(min_width)] for char in chars]
    # Sort font bitmaps by pixel density.
    fonts.sort(key=lambda x: x.sum(), reverse=not reverse)
    return np.array(fonts)


def draw_ascii(frame, chars, background, clip, monochrome, font_bitmaps, buffer=None):
    """
    Draws an ASCII Image.

    Parameters
        frame           - Numpy array representing image
        chars           - ASCII characters to use in media
        background      - Background color
        clip            - Clip characters to not go outside of image bounds
        monochrome      - Color to use for monochromatic. None if not monochromatic
        font_bitmaps    - List of font bitmaps
        buffer          - Optional buffer for intermediary calculations

    NOTE: Characters such as q, g, y, etc... are not rendered properly in this implementation
    due to the lower ends being cut off.
    """
    if buffer is None:
        buffer = np.empty_like(frame, dtype=np.uint16 if len(chars) < 32 else np.uint32)

    # fh -> font height, fw -> font width.
    fh, fw = font_bitmaps[0].shape[:2]
    # oh -> Original height, ow -> Original width.
    oh, ow = frame.shape[:2]
    # Sample original frame at steps of font width and height.
    frame = frame[::fh, ::fw]
    h, w = frame.shape[:2]

    buffer_view = buffer[:h, :w]
    if len(monochrome) != 0:
        buffer_view[:] = 1
        if background == 255:
            monochrome = 255 - monochrome
        np.multiply(buffer_view, monochrome, out=buffer_view)
    else:
        if background == 255:
            np.subtract(255, frame, out=buffer_view)
        else:
            buffer_view[:] = frame
    
    colors = buffer_view.repeat(fw, 1).astype(np.uint16, copy=False).repeat(fh, 0)

    # Grayscale original frame and normalize to ASCII index.
    buffer_view = buffer_view[..., 0]
    np.sum(frame * np.array([3, 4, 1]), axis=2, dtype=buffer.dtype, out=buffer_view)
    buffer_view *= len(chars)
    buffer_view >>= 11

    # Create a new list with each font bitmap based on the grayscale value.
    image = (
        font_bitmaps[buffer_view.reshape(-1)]
        .reshape((h, w, fh, fw, 3))
        .transpose(0, 2, 1, 3, 4)
        .reshape((h * fh, w * fw, 3))
    )

    if clip:
        colors = colors[:oh, :ow]
        image = image[:oh, :ow]

    np.multiply(image, colors, out=buffer)
    np.floor_divide(buffer, 255, out=buffer)
    buffer = buffer.astype(np.uint8, copy=False)
    if background == 255:
        np.subtract(255, buffer, out=buffer)
    return buffer


def ascii_video(
    filename,
    output,
    chars,
    monochrome,
    fontsize=20,
    boldness=2,
    reverse=False,
    background=255,
    clip=True,
    font="cour.ttf",
    audio=False,
):
    font_bitmaps = get_font_bitmaps(fontsize, boldness, reverse, background, chars, font)

    video = imageio_ffmpeg.read_frames(filename)
    data = next(video)

    w, h = data["size"]
    frame_size = (h, w, 3)
    # Smaller data types can speed up operations. The minimum data type required will be
    # 2^n / (255 * 8) > len(chars) where n = 16 or 32.
    buffer = np.empty(frame_size, dtype=np.uint16 if len(chars) < 32 else np.uint32)
    # Read and convert first frame to figure out frame size.
    first_frame = np.frombuffer(next(video), dtype=np.uint8).reshape(frame_size)
    first_frame = draw_ascii(first_frame, chars, background, clip, monochrome, font_bitmaps, buffer)
    h, w = first_frame.shape[:2]

    kwargs = {"fps": data["fps"]}
    if audio:
        kwargs["audio_path"] = filename

    writer = imageio_ffmpeg.write_frames(output, (w, h), **kwargs)
    writer.send(None)
    writer.send(first_frame)

    for frame in ProgressBar(video, total=int(data["fps"] * data["duration"] - 0.5)):
        frame = np.frombuffer(frame, dtype=np.uint8).reshape(frame_size)
        writer.send(draw_ascii(frame, chars, background, clip, monochrome, font_bitmaps, buffer))

    writer.close()


def ascii_image(
    filename,
    output,
    chars,
    monochrome,
    fontsize=20,
    boldness=2,
    reverse=False,
    background=255,
    clip=True,
    font="cour.ttf",
):
    image = imageio.imread(filename)[:, :, :3]
    font_bitmaps = get_font_bitmaps(fontsize, boldness, reverse, background, chars, font)
    image = draw_ascii(image, chars, background, clip, monochrome, font_bitmaps)
    imageio.imsave(output, image)


def parse_args():
    parser = argparse.ArgumentParser(description="Blazing fast ASCII Media converter.")

    parser.add_argument("filename", help="File name of the input image.")
    parser.add_argument("output", help="File name of the output image.")

    parser.add_argument("-chars", "--characters", help="ASCII chars to use in media.", default="@%#*+=-:. ")
    parser.add_argument("-r", "--reverse", help="Reverse the character order.", action="store_true")
    parser.add_argument("-f", "--fontsize", help="Font size.", type=int, default=20)
    parser.add_argument("-b", "--bold", help="Boldness of characters. Recommended: 1/10 font size.", type=int, default=2)
    parser.add_argument("-bg", "--background", help="Background color. Must be 255 (white) or 0 (black).", type=int, default=255)
    parser.add_argument("-m", "--monochrome", help='Color to use for Monochromatic characters in "R,G,B" format.')
    parser.add_argument("-c", "--clip", help="Clip characters to not go outside of image bounds.", action="store_false")
    parser.add_argument("-font", "--font", help="Font to use.", type=str, default="cour.ttf")
    parser.add_argument("-a", "--audio", help="Add audio from the input file to the output file.", action="store_true")

    return parser.parse_args()


def convert_ascii(args, filename, output, chars, monochrome):
    try:
        imageio.imread(filename)
    except Exception:
        ascii_video(
            filename,
            output,
            chars,
            monochrome,
            args.fontsize,
            args.bold,
            args.reverse,
            args.background,
            args.clip,
            args.font,
            args.audio,
        )
    else:
        ascii_image(
            filename,
            output,
            chars,
            monochrome,
            args.fontsize,
            args.bold,
            args.reverse,
            args.background,
            args.clip,
            args.font,
        )


def main():
    args = parse_args()

    assert args.fontsize > 0, "Font size must be > 0."
    assert args.bold >= 0, "Boldness must be >= 0."
    assert args.background in [0, 255], "Background must be either 0 or 255."

    chars = np.array(list(args.characters))
    monochrome = np.array(
        list(map(int, args.monochrome.split(","))) if args.monochrome else [],
        dtype=np.uint16,
    )

    if os.path.isdir(args.filename):
        os.makedirs(args.output, exist_ok=True)
        for filename in ProgressBar(os.listdir(args.filename)):
            path = os.path.join(args.filename, filename)
            output = os.path.join(args.output, filename)
            convert_ascii(args, path, output, chars, monochrome)
    else:
        convert_ascii(args, args.filename, args.output, chars, monochrome)


if __name__ == "__main__":
    main()
