# ASCII Video Converter

Convert Video and Images to ASCII form!

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

## Dependencies

* Python 3.7+
* `imageio`
* `numpy`
* `PIL`
* `tqdm`

```
pip install numpy pillow tqdm imageio
pip install imageio-ffmpeg --user
```


# Images

### Butterfly

<img src="Documentation/butterfly.gif" alt="Butterfly ASCII">

### Oreo

<img src="Documentation/oreo.gif" alt="Oreo Cookie">

<br /><br />

# Video

### Original

<img src="Documentation/original.gif" alt="Donuts">

### Color

<img src="Documentation/donuts.gif" alt="ASCII Donuts">

### Monochrome

<img src="Documentation/donuts-mono.gif" alt="ASCII Donuts Monochrome">