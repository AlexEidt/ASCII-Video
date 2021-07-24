"""
Alex Eidt

Converts videos/images into ASCII video/images in color, grayscale and monochrome.
"""

import argparse
import imageio
import numpy as np
import multiprocessing
from tqdm import tqdm
from typing import Union, Tuple, Callable
from PIL import Image, ImageFont, ImageDraw


# Dictionary storing mapping of font sizes to tuples of font ttfs and sizes.
FONTS = {i: ImageFont.truetype('cour.ttf', size=i) for i in range(1, 100)}
FONTS = {i: (font, (font.getsize('K'))) for i, font in FONTS.items()}

BACKGROUND_COLOR = 255
assert BACKGROUND_COLOR in (255, 0), 'BACKGROUND_COLOR must be either 0 or 255'


def get_font_maps(
    fontsize:       int,
    boldness:       int,
    chars:          str,
) -> list:
    """
    Returns a list of font bitmaps.

    Parameters
        fontsize    - Font size to use for ASCII characters
        boldness    - Stroke size to use when drawing ASCII characters
        chars       - ASCII characters to use in media

    Returns
        List of font bitmaps corresponding to the order of characters in CHARS
    """
    fonts = []
    widths, heights = set(), set()
    font, _ = FONTS[fontsize]
    for char in chars:
        w, h = font.getsize(char)
        widths.add(w)
        heights.add(h)
        image = Image.new("RGB", (w, h), (BACKGROUND_COLOR,) * 3)
        draw = ImageDraw.Draw(image)
        draw.text(
            (0, - (fontsize // 6)),
            char,
            fill=(255 - BACKGROUND_COLOR,) * 3,
            font=font,
            stroke_width=boldness
        )
        bitmap = np.array(image)[:, :, 0]
        if BACKGROUND_COLOR == 255:
            bitmap = 255 - bitmap
        fonts.append((bitmap / 255).astype(np.float32))

    return list(map(lambda x: x[:min(heights), :min(widths)], fonts))


def draw(
    params: Tuple[
        np.array,
        str,
        int,
        int,
        bool,
        bool,
        Union[Tuple[int, int, int], None],
        None
    ]
) -> np.array:
    """
    Draws an ASCII Image.

    Parameters
        params - A tuple holding 3 parameters: frame, fontsize, bold.
            frame       - Numpy array representing image
            chars       - ASCII characters to use in media
            fontsize    - Font size to use for ASCII characters
            boldness    - Stroke size to use when drawing ASCII characters
            clip        - Clip characters to not go outside of image bounds
            monochrome  - Color to use for monochromatic. None if not monochromatic
            font_maps   - For use with "draw_efficient". None is passed for "draw".
    Returns
        Numpy array representing ASCII Image
    """
    frame, chars, fontsize, boldness, clip, monochrome, _ = params
    global FONTS
    font, (fw, fh) = FONTS[fontsize]
    grayscaled = np.sum(
        frame * np.array([0.299, 0.587, 0.114]),
        axis=2,
        dtype=np.uint32
    ) * (len(chars) - 1) // 255
    # Convert to ascii index
    ascii_map = np.vectorize(lambda x: chars[x])(grayscaled)
    h, w = grayscaled.shape

    if clip:
        h = (h // fh) * fh - fh
        w = (w // fw) * fw - fw

    image = Image.new("RGB", (w, h), (BACKGROUND_COLOR,) * 3)
    draw = ImageDraw.Draw(image)

    for row in np.arange(0, h, fh):
        for column in np.arange(0, w, fw):
            draw.text(
                (column, row),
                ascii_map[int(row), int(column)],
                fill=tuple(frame[int(row), int(column)]) if monochrome is None else monochrome,
                font=font,
                stroke_width=boldness
            )

    return np.array(image)


def draw_efficient(
    params: Tuple[
        np.array,
        str,
        int,
        int,
        bool,
        Union[Tuple[int, int, int], None],
        list
    ]
) -> np.array:
    """
    Draws an ASCII Image. This function is heavily optimized, achieving around a 100x
    speedup over the "draw" function, with some drawbacks. Hanging characters such as q, g, y, etc...
    are not rendered properly in this implementation due to the lower ends being cut off.
    Characters are also shifted to adjust for a bug in the ImageFont library which clips
    rendered characters. The user may choose to discard all characters with hanging
    components to utilize this function without much drawback.

    Parameters
        params - A tuple holding 3 parameters: frame, fontsize, bold.
            frame       - Numpy array representing image
            chars       - ASCII characters to use in media
            fontsize    - Font size to use for ASCII characters
            boldness    - Stroke size to use when drawing ASCII characters
            clip        - Clip characters to not go outside of image bounds
            monochrome  - Color to use for monochromatic. None if not monochromatic
            font_maps   - List of font bitmaps

    Returns
        Numpy array representing ASCII Image
    """
    frame, chars, fontsize, boldness, clip, monochrome, font_maps = params
    fh, fw = font_maps[0].shape
    oh, ow, _ = frame.shape
    frame = frame[::fh, ::fw]
    h, w, _ = frame.shape

    bg_white = BACKGROUND_COLOR == 255
    if monochrome is not None:
        colors = np.array(monochrome)
        if bg_white:
            colors = 255 - colors
    elif bg_white:
        colors = np.repeat(np.repeat(255 - frame, fw, axis=1), fh, axis=0)
    else:
        colors = np.repeat(np.repeat(frame, fw, axis=1), fh, axis=0)
    grayscaled = np.sum(
        frame * np.array([0.299, 0.587, 0.114]),
        axis=2,
        dtype=np.uint32
    ).ravel() * (len(chars) - 1) // 255

    # Create a new list with each font bitmap based on the grayscale value
    image = map(lambda idx: font_maps[grayscaled[idx]], range(len(grayscaled)))
    image = np.array(list(image)).reshape((h, w, fh, fw)).transpose(0, 2, 1, 3).ravel()
    image = np.tile(image, 3).reshape((3, h * fh, w * fw)).transpose(1, 2, 0)
    if clip:
        if monochrome is None:
            colors = colors[:oh, :ow]
        image = (image[:oh, :ow] * colors).astype(np.uint8)
        if bg_white:
            return 255 - image
        return image
    image = (image * colors).astype(np.uint8)
    if bg_white:
        return 255 - image
    return image


def asciify(
    filename:   str,
    output:     str,
    chars:      str,
    fontsize:   int,
    boldness:   int,
    font_maps:  list,
    cores:      int,
    draw_func:  Callable,
    monochrome: Union[Tuple[int, int, int], None] = None,
    clip:       bool = True,
    random:     bool = False,
    width:      int = 1920,
    height:     int = 1088,
    fps:        Union[int, float] = 30,
    duration:   Union[int, float] = 10,
) -> None:
    """
    Converts a given video into an ASCII video.

    Parameters
        filename    - Name of the input video file
        output      - Name of the output video file
        chars       - ASCII characters to use in media
        fontsize    - Font size to use for ASCII characters
        boldness    - Stroke size to use when drawing ASCII characters
        font_maps   - List of font bitmaps
        cores       - CPU Cores to use when processing images
        draw_func   - Drawing function to use
        monochrome  - Color to use for Monochromatic characters, otherwise None
        clip        - Clip characters to not go outside of image bounds
        random      - If True, create random video, otherwise use given filename
        width       - Width of video (For use with random=True only)
        height      - Height of video (For use with random=True only)
        fps         - Frames per second of randomized video (For use with random=True only)
        duration    - Duration (in seconds) of randomized video (For use with random=True only)
    """
    if random:
        data = {'fps': fps, 'duration': duration, 'source_size': (width, height)}
        frames = (
            np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            for _ in range(int(fps * duration + 0.5))
        )
    else:
        video = imageio.get_reader(filename)
        data = video.get_meta_data()
        frames = iter(video)

    length = int(data['fps'] * data['duration'] + 0.5)
    w, h = data['source_size']
    with imageio.get_writer(output, fps=data['fps']) as writer:
        if cores <= 1 or draw_func.__name__ == 'draw_efficient':
            for frame in tqdm(frames, total=length):
                writer.append_data(
                    draw_func((frame, chars, fontsize, boldness, clip, monochrome, font_maps))
                )
        else:
            progress_bar = tqdm(total=int(length / cores + 0.5))
            while True:
                batch = []
                for _ in range(cores):
                    try:
                        frame = next(frames)
                    except StopIteration:
                        break
                    else:
                        batch.append((frame, chars, fontsize, boldness, clip, monochrome, None))

                if batch:
                    with multiprocessing.Pool(processes=len(batch)) as pool:
                        for ascii_frame in pool.map(draw, batch):
                            writer.append_data(ascii_frame)
                    progress_bar.update()
                else:
                    break
    if not random:
        video.close()


def ascii_image(
    filename:   str,
    output:     str,
    chars:      str,
    fontsize:   int,
    boldness:   int,
    font_maps:  list,
    draw_func:  Callable,
    monochrome: Union[Tuple[int, int, int], None] = None,
    clip:       bool = True,
    random:     bool = False,
    width:      int = 1920,
    height:     int = 1088
) -> None:
    """
    Converts an image into an ASCII Image.

    Parameters
        filename    - File name of the input image
        output      - File name of the output image
        chars       - ASCII characters to use in media
        fontsize    - Font size to use for ASCII characters
        boldness    - Stroke size to use when drawing ASCII characters
        font_maps   - List of font bitmaps
        draw_func   - Drawing function to use
        monochrome  - Color to use for Monochromatic characters, otherwise None
        clip        - Clip characters to not go outside of image bounds
        random      - If True, create random image, otherwise use given filename
        width       - Width of video
        height      - Height of video
    """
    if random:
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        h, w = height, width
    else:
        image = imageio.imread(filename)[:, :, :3]
        h, w, _ = image.shape
    font_maps = get_font_maps(fontsize, boldness, chars) if draw_func.__name__ == 'draw_efficient' else None
    image = draw_func((image, chars, fontsize, boldness, clip, monochrome, font_maps))
    imageio.imsave(output, image)


def main():
    parser = argparse.ArgumentParser(description='Blazing fast ASCII Media converter.')
    parser.add_argument('filename', help='File name of the input image.')
    parser.add_argument('output', help='File name of the output image.')

    parser.add_argument('-chars', required=False, help='ASCII chars to use in media.')
    parser.add_argument('-f', required=False, help='Font size.', nargs='?', const=1, type=int, default=20)
    parser.add_argument('-b', required=False, help='Boldness of characters. Recommended boldness is 1/10 of Font size.', nargs='?', const=1, type=int, default=2)
    parser.add_argument('-d', required=False, help="Use normal drawing algorithm over efficient one.", action='store_true')
    parser.add_argument('-m', required=False, help='Color to use for Monochromatic characters in R, G, B format.')
    parser.add_argument('-c', required=False, help='Clip characters to not go outside of image bounds.', action='store_true')
    parser.add_argument('-r', required=False, help='Draw random ASCII characters.', action='store_true')
    parser.add_argument('-height', required=False, help='Height of random ASCII media.', nargs='?', const=1, type=int, default=1080)
    parser.add_argument('-width', required=False, help='Width of random ASCII media.', nargs='?', const=1, type=int, default=1920)
    parser.add_argument('-cores', required=False, help='CPU Cores to use when processing images.', nargs='?', const=1, type=int, default=4)
    parser.add_argument('-fps', required=False, help='Frames per second of randomized video (For use with random only).', nargs='?', const=1, type=int, default=30)
    parser.add_argument('-dur', required=False, help='Duration (in seconds) of randomized video (For use with random only)', nargs='?', const=1, type=int, default=10)

    args = parser.parse_args()

    chars = f""" `.,|'\\/~!_-;:)(\"><?*+7j1ilJyc&vt0$VruoI=wzCnY32LTxs4Zkm5hg6qfU9paOS#eX8D%bdRPGFK@AMQNWHEB"""[::-1]
    filename = args.filename
    output = args.output
    chars = ''.join([c for c in args.chars if c in chars]) if args.chars else chars
    monochrome = tuple(map(int, args.m.split(','))) if args.m else None
    font_maps = get_font_maps(args.f, args.b, chars)
    cores = min(args.cores, multiprocessing.cpu_count())
    if filename.endswith(('png', 'jpg', 'jpeg')):
        ascii_image(
            filename,
            output,
            chars,
            args.f,
            args.b,
            font_maps,
            draw_efficient if not args.d else draw,
            monochrome,
            args.c,
            args.r,
            args.width,
            args.height
        )
    else:
        asciify(
            filename,
            output,
            chars,
            args.f,
            args.b,
            font_maps,
            cores,
            draw_efficient if not args.d else draw,
            monochrome,
            args.c,
            args.r,
            args.width,
            args.height,
            args.fps,
            args.dur
        )


if __name__ == '__main__':
    main()