# ASCII Video Converter

Convert Video and Images to ASCII form! Achieve Real-Time Color ASCII Rendering using NumPy Vectorization.

See this algorithm work on a webcam stream: https://github.com/AlexEidt/ASCII-Vision

## Usage

```
positional arguments:
  filename      File name of the input image.
  output        File name of the output image.

optional arguments:
  -h, --help    show this help message and exit
  -chars CHARS  ASCII chars to use in media.
  -f [F]        Font size.
  -b [B]        Boldness of characters. Recommended boldness is 1/10 of Font size.
  -bg [BG]      Background color. Must be either 255 for white or 0 for black.
  -m M          Color to use for Monochromatic characters in "R,G,B" format.
  -c            Clip characters to not go outside of image bounds.
  -font [FONT]  Font to use.
  -a            Add audio from the input file to the output file.
```

Instead of a filename, a directory name can be used as the `filename` argument. All media in this directory will be converted to ASCII Form and be placed in a directory specified by the `output` argument. Note that this means if a directory is passed as the `filename` argument, the `output` argument must also be a directory and all media in `filename` must be either images and/or videos.

## Dependencies

* Python 3.7+
* `imageio`
* `imageio-ffmpeg`
* `numpy`
* `PIL`
* `tqdm`

```
pip install numpy pillow tqdm imageio imageio-ffmpeg
```

# Images

<img src="Documentation/butterfly.jpg" alt="Butterfly" />

<img src="Documentation/butterfly-ascii-color.png" alt="Butterfly ASCII Color" />

<img src="Documentation/butterfly-ascii-mono.png" alt="Butterfly ASCII Monochrome" />


<img src="Documentation/houses.jpg" alt="Houses" />

<img src="Documentation/houses-ascii-color.png" alt="Houses ASCII Color" />

<img src="Documentation/houses-ascii-mono.png" alt="Houses ASCII Monochrome" />


# Video

<img src="Documentation/donuts.gif" alt="Donuts">

<img src="Documentation/donuts-ascii.gif" alt="Donuts ASCII">
