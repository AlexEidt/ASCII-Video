"""
Alex Eidt

Converts videos/images into ASCII video/images in various formats.

A simple character set to use: "@%#*+=-:. "

"""

import argparse
import string
import imageio
import numpy as np
from tqdm import tqdm as ProgressBar
from PIL import Image, ImageFont, ImageDraw


def get_font_maps(fontsize, boldness, background, chars, font):
    """
    Returns a list of font bitmaps.

    Parameters
        fontsize    - Font size to use for ASCII characters
        boldness    - Stroke size to use when drawing ASCII characters
        background  - Background color
        chars       - ASCII characters to use in media
        font        - Font to use

    Returns
        List of font bitmaps corresponding to the characters in "chars".
    """
    fonts = []
    min_width = min_height = float("inf")
    font_ttf = ImageFont.truetype(font, size=fontsize)

    for char in chars:
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
            stroke_width=boldness
        )
        # Since font bitmaps are grayscale, all three color channels contain the same
        # information, so one channel is extracted.
        bitmap = np.array(image)[:, :, 0]
        if background == 255:
            bitmap = 255 - bitmap
        fonts.append((bitmap / 255).astype(np.float32))

    # Crop the font bitmaps to all have the same dimensions based on the
    # minimum font width and height of all font bitmaps.
    fonts = [bitmap[:int(min_height), :int(min_width)] for bitmap in fonts]
    # Sort font bitmaps by pixel density.
    return np.array(sorted(fonts, key=lambda x: x.sum(), reverse=True))


def draw_ascii(frame, chars, background, clip, monochrome, font_maps):
    """
    Draws an ASCII Image.

    Parameters
        frame       - Numpy array representing image
        chars       - ASCII characters to use in media
        background  - Background color
        clip        - Clip characters to not go outside of image bounds
        monochrome  - Color to use for monochromatic. None if not monochromatic
        font_maps   - List of font bitmaps
    
    NOTE: Characters such as q, g, y, etc... are not rendered properly in this implementation
    due to the lower ends being cut off.
    """
    # fh -> font height.
    # fw -> font width.
    fh, fw = font_maps[0].shape
    # oh -> Original height.
    # ow -> Original width.
    oh, ow = frame.shape[:2]
    # Sample original frame at steps of font width and height.
    frame = frame[::fh, ::fw]
    h, w = frame.shape[:2]

    if len(monochrome):
        colors = 255 - monochrome if background == 255 else monochrome
    else:
        colors = np.repeat(
            np.repeat(255 - frame if background == 255 else frame, fw, axis=1), fh, axis=0
        )

    # Grayscale original frame and normalize to ASCII index.
    frame = np.sum(frame * np.array([3, 4, 1]), axis=2, dtype=np.uint32).ravel()
    frame *= len(chars)
    frame >>= 11

    # Create a new list with each font bitmap based on the grayscale value.
    frame = frame[range(len(frame))]
    image = font_maps[frame]
    image = image.reshape((h, w, fh, fw)).transpose(0, 2, 1, 3).ravel()
    image = np.tile(image, 3).reshape((3, h * fh, w * fw)).transpose(1, 2, 0)

    if clip:
        if not len(monochrome):
            colors = colors[:oh, :ow]
        image = image[:oh, :ow]

    image = (image * colors).astype(np.uint8)
    return 255 - image if background == 255 else image


def ascii_video(
    filename,
    output,
    chars,
    monochrome,
    fontsize=20,
    boldness=2,
    background=255,
    clip=True,
    font='cour.ttf'
):
    font_maps = get_font_maps(fontsize, boldness, background, chars, font)

    with imageio.read(filename) as video:
        data = video.get_meta_data()

        with imageio.save(output, fps=data['fps']) as writer:
            for frame in ProgressBar(video, total=int(data['fps'] * data['duration'] + 0.5)):
                writer.append_data(draw_ascii(frame, chars, background, clip, monochrome, font_maps))


def ascii_image(
    filename,
    output,
    chars,
    monochrome,
    fontsize=20,
    boldness=2,
    background=255,
    clip=True,
    font='cour.ttf'
):
    image = imageio.imread(filename)[:, :, :3]
    font_maps = get_font_maps(fontsize, boldness, background, chars, font)
    image = draw_ascii(image, chars, background, clip, monochrome, font_maps)
    imageio.imsave(output, image)


def main():
    parser = argparse.ArgumentParser(description='Blazing fast ASCII Media converter.')

    parser.add_argument('filename', help='File name of the input image.')
    parser.add_argument('output', help='File name of the output image.')

    parser.add_argument('-chars', required=False, help='ASCII chars to use in media.', default=string.printable)
    parser.add_argument('-f', required=False, help='Font size.', nargs='?', const=1, type=int, default=20)
    parser.add_argument('-b', required=False, help='Boldness of characters. Recommended boldness is 1/10 of Font size.', nargs='?', const=1, type=int, default=2)
    parser.add_argument('-bg', required=False, help='Background color. Must be either 255 for white or 0 for black.', nargs='?', const=1, type=int, default=255)
    parser.add_argument('-m', required=False, help='Color to use for Monochromatic characters in "R,G,B" format.')
    parser.add_argument('-c', required=False, help='Clip characters to not go outside of image bounds.', action='store_false')
    parser.add_argument('-font', required=False, help='Font to use.', nargs='?', const=1, type=str, default='cour.ttf')

    args = parser.parse_args()

    chars = np.array([c for c in string.printable if c in args.chars])
    monochrome = np.array(list(map(int, args.m.split(','))) if args.m else [])

    try:
        imageio.imread(args.filename)
    except Exception:
        ascii_video(
            args.filename, args.output, chars, monochrome,
            args.f, args.b, args.bg,
            args.c,
            args.font
        )
    else:
        ascii_image(
            args.filename, args.output, chars, monochrome,
            args.f, args.b, args.bg,
            args.c,
            args.font
        )


if __name__ == '__main__':
    main()