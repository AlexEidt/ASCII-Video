"""
Alex Eidt

Converts videos/images into ASCII video/images in color, grayscale and monochrome.
"""

import argparse
import imageio
import numpy as np
import multiprocessing
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw


# CPU Cores to use when processing frames.
CORES = min(4, multiprocessing.cpu_count())
# If True, create Grayscaled images/videos.
GRAY = False
# If True, use monochrome (all black) ASCII characters only.
MONOCHROME = False
# Set of ASCII Characters to use. Sorted by pixel density.
CHARS = f""" `.,|'\\/~!_-;:)(\"><?*+7j1ilJyc&vt0$VruoI=wzCnY32LTxs4Zkm5hg6qfU9paOS#eX8D%bdRPGFK@AMQNWHEB"""[::-1]
# Dictionary storing mapping of font sizes to tuples of font ttfs and sizes.
FONTS = {i: ImageFont.truetype('cour.ttf', size=i) for i in range(1, 100)}
FONTS = {i: (font, (font.getsize("K"))) for i, font in FONTS.items()}


def draw(params):
    """
    Draws an ASCII Image.

    Parameters:
        params - A tuple holding 3 parameters: frame, fontsize, bold.
            frame    - Numpy array representing image
            fontsize - Font size to use for ASCII characters
            bold     - Stroke size to use when drawing ASCII characters

    Returns
        Numpy array representing ASCII Image
    """
    frame, fontsize, bold = params
    font, (fw, fh) = FONTS[fontsize]
    grayscaled = np.sum(frame * np.array([0.299, 0.587, 0.114]), axis=2, dtype=np.uint16)
    # Convert to ascii index
    ascii_map = np.vectorize(lambda x: CHARS[x])((grayscaled * (len(CHARS) - 1)) // 255)
    if GRAY:
        frame = np.dstack([grayscaled] * 3)
    elif MONOCHROME:
        frame = np.zeros(frame.shape, dtype=np.uint8)
    h, w = grayscaled.shape

    image = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    for row in np.arange(0, h, fh):
        for column in np.arange(0, w, fw):
            draw.text(
                (column, row),
                ascii_map[int(row), int(column)],
                fill=tuple(frame[int(row), int(column)]),
                font=font,
                stroke_width=bold
            )

    return np.array(image)


def asciify(filename, output, fontsize, boldness):
    """
    Converts a given video into an ASCII video.

    Parameters
        filename - Name of the input video file
        output   - Name of the output video file
        fontsize - Font size to use for ASCII characters
        boldness - Stroke size to use when drawing ASCII characters
    """
    with imageio.get_reader(filename) as video:
        data = video.get_meta_data()
        length = int(data['fps'] * data['duration'] + 0.5)

        with imageio.get_writer(output, fps=data['fps']) as writer:
            size = int(length / CORES + 0.5)
            if CORES <= 1:
                for i, frame in tqdm(enumerate(video), total=size):
                    writer.append_data(draw((frame, fontsize, boldness)))
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
                            batch.append((frame, fontsize, boldness))

                    if batch:
                        with multiprocessing.Pool(processes=len(batch)) as pool:
                            for result in pool.map(draw, batch):
                                writer.append_data(result)
                        progress_bar.update()
                    else:
                        break


def ascii_image(filename, output, fontsize, boldness, random=False, width=1920, height=1088):
    """
    Converts an image into an ASCII Image.

    Parameters
        filename - File name of the input image
        output   - File name of output image
        fontsize - Font size to use for ASCII characters
        boldness - Stroke size to use when drawing ASCII characters
        random   - If True, create random image, otherwise use given filename
        width    - Width of video
        height   - Height of video
    """
    if random:
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    else:
        image = imageio.imread(filename)[:, :, :3]
    image = draw((image, fontsize, boldness))
    imageio.imsave(output, image)


def random_ascii(filename, fps, duration, fontsize, boldness, width=1920, height=1088):
    """
    Creates a video with random characters and colors.

    Parameters
        filename - Name of the output video file
        fps      - Frames per second of video
        duration - Duration (in seconds) of video
        fontsize - Font size to use for ASCII characters
        boldness - Stroke size to use when drawing ASCII characters
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
                    batch.append((random_frame, fontsize, boldness))

                with multiprocessing.Pool(processes=len(batch)) as pool:
                    for result in pool.map(draw, batch):
                        writer.append_data(result)


def main():
    fontsize = 20
    boldness = 2 # Recommended Boldness is Font Size divided by 10

    # Example Function Calls

    # asciify(filename, output, fontsize, boldness)
    # ascii_image(filename, output, fontsize, boldness, random=False, width=1920, height=1088)
    # random_ascii(filename, fps, duration, fontsize, boldness, width=1920, height=1088)


if __name__ == '__main__':
    main()