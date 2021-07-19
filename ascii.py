"""
Alex Eidt

Converts videos/images into ASCII video/images in color, grayscale and monochrome.
"""

import imageio
import numpy as np
import multiprocessing
from tqdm import tqdm
from typing import Union, Tuple
from PIL import Image, ImageFont, ImageDraw


# CPU Cores to use when processing frames.
CORES = min(4, multiprocessing.cpu_count())
# If True, create Grayscaled images/videos.
GRAY = False
# If True, use monochrome (all black) ASCII characters only.
MONOCHROME = False
# Set of ASCII Characters to use. Sorted by pixel density.
# Feel free to change/customize which ASCII characters to use.
CHARS = f""" `.,|'\\/~!_-;:)(\"><?*+7j1ilJyc&vt0$VruoI=wzCnY32LTxs4Zkm5hg6qfU9paOS#eX8D%bdRPGFK@AMQNWHEB"""[::-1]
# Change ASCII Characters used via the 'CUSTOM_CHARS' variable
CUSTOM_CHARS = '@%#*+=-:. '
# Uncomment the following line if using custom characters
# CHARS = ''.join([x for x in CHARS if x in CUSTOM_CHARS])

# Dictionary storing mapping of font sizes to tuples of font ttfs and sizes.
FONTS = {i: ImageFont.truetype('cour.ttf', size=i) for i in range(1, 100)}
FONTS = {i: (font, (font.getsize('K'))) for i, font in FONTS.items()}
# Background color to use. Default is white.
BACKGROUND_COLOR = (255, 255, 255)


def draw(params: Tuple[np.array, int, int, bool]) -> np.array:
    """
    Draws an ASCII Image.

    Parameters:
        params - A tuple holding 3 parameters: frame, fontsize, bold.
            frame    - Numpy array representing image
            fontsize - Font size to use for ASCII characters
            bold     - Stroke size to use when drawing ASCII characters
            clip     - Clip characters to not go outside of image bounds

    Returns
        Numpy array representing ASCII Image
    """
    frame, fontsize, boldness, clip = params
    font, (fw, fh) = FONTS[fontsize]
    grayscaled = np.sum(frame * np.array([0.299, 0.587, 0.114]), axis=2, dtype=np.uint16)
    # Convert to ascii index
    ascii_map = np.vectorize(lambda x: CHARS[x])((grayscaled * (len(CHARS) - 1)) // 255)
    if GRAY:
        frame = np.dstack([grayscaled] * 3)
    elif MONOCHROME:
        frame = np.zeros(frame.shape, dtype=np.uint8)
    h, w = grayscaled.shape

    if clip:
        h = (h // fh) * fh - fh
        w = (w // fw) * fw - fw

    image = Image.new("RGB", (w, h), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(image)

    for row in np.arange(0, h, fh):
        for column in np.arange(0, w, fw):
            draw.text(
                (column, row),
                ascii_map[int(row), int(column)],
                fill=tuple(frame[int(row), int(column)]),
                font=font,
                stroke_width=boldness
            )

    return np.array(image)


def asciify(
    filename: str,
    output: str,
    fontsize: int,
    boldness: int,
    clip: bool = False
) -> None:
    """
    Converts a given video into an ASCII video.

    Parameters
        filename - Name of the input video file
        output   - Name of the output video file
        fontsize - Font size to use for ASCII characters
        boldness - Stroke size to use when drawing ASCII characters
        clip     - Clip characters to not go outside of image bounds
    """
    with imageio.get_reader(filename) as video:
        data = video.get_meta_data()
        length = int(data['fps'] * data['duration'] + 0.5)

        with imageio.get_writer(output, fps=data['fps']) as writer:
            size = int(length / CORES + 0.5)
            if CORES <= 1:
                for frame in tqdm(video, total=size):
                    writer.append_data(draw((frame, fontsize, boldness, clip)))
            else:
                video = iter(video)
                progress_bar = tqdm(total=size)
                while True:
                    batch = []
                    for _ in range(CORES):
                        try:
                            frame = next(video)
                        except StopIteration:
                            break
                        else:
                            batch.append((frame, fontsize, boldness, clip))

                    if batch:
                        with multiprocessing.Pool(processes=len(batch)) as pool:
                            for ascii_frame in pool.map(draw, batch):
                                writer.append_data(ascii_frame)
                        progress_bar.update()
                    else:
                        break


def ascii_image(
    filename:   str,
    output:     str,
    fontsize:   int,
    boldness:   int,
    clip:       bool = False,
    random:     bool = False,
    width:      int = 1920,
    height:     int = 1088
) -> None:
    """
    Converts an image into an ASCII Image.

    Parameters
        filename - File name of the input image
        output   - File name of output image
        fontsize - Font size to use for ASCII characters
        boldness - Stroke size to use when drawing ASCII characters
        clip     - Clip characters to not go outside of image bounds
        random   - If True, create random image, otherwise use given filename
        width    - Width of video
        height   - Height of video
    """
    if random:
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    else:
        image = imageio.imread(filename)[:, :, :3]
    image = draw((image, fontsize, boldness, clip))
    imageio.imsave(output, image)


def random_ascii(
    filename:   str,
    fps:        Union[int, float],
    duration:   Union[int, float],
    fontsize:   int,
    boldness:   int,
    clip:       bool = False,
    width:      int = 1920,
    height:     int = 1088
) -> None:
    """
    Creates a video with random characters and colors.

    Parameters
        filename - Name of the output video file
        fps      - Frames per second of video
        duration - Duration (in seconds) of video
        fontsize - Font size to use for ASCII characters
        boldness - Stroke size to use when drawing ASCII characters
        clip     - Clip characters to not go outside of image bounds
        width    - Width of video
        height   - Height of video
    """
    with imageio.get_writer(filename, fps=fps) as writer:
        size = np.arange(int(fps * duration))
        size = [size[i:i+CORES] for i in range(0, len(size), CORES)]
        for indices in tqdm(size):
            if len(indices):
                batch = []
                for _ in range(len(indices)):
                    random_frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                    batch.append((random_frame, fontsize, boldness, clip))

                with multiprocessing.Pool(processes=len(batch)) as pool:
                    for ascii_frame in pool.map(draw, batch):
                        writer.append_data(ascii_frame)


def main():
    fontsize = 20
    boldness = 2 # Recommended Boldness is Font Size divided by 10

    # Example Function Calls

    # asciify(filename, output, fontsize, boldness, clip=False)
    # ascii_image(filename, output, fontsize, boldness, clip=False, random=False, width=1920, height=1088)
    # random_ascii(filename, fps, duration, fontsize, boldness, clip=False, width=1920, height=1088)


if __name__ == '__main__':
    main()