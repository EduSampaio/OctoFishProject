# Stereo calibration:

This script is designed to calibrate a stereo camera and save the
required calibration metrix in a ouput (.npz) file. Further verbosity of
the viewable output can be enabled by setting the equivalent flag (see
below).

## Usage

Reliably tested with these versions:

|Dependencies|Version |
|:---|:---:|
|Python|3.7.0 |
|OpenCV|3.4.3|
|numpy|1.15.2|
|matplotlib|2.2.3|

In order to use the stereo calibration script (command line interface)
navigate to the
containing folder and execute
```
python stereo_calib.py
```
followed by the additional arguments for specification:

|Argument       | Use           |Default |
|:-------------: |:-------------| :-----|
|-r|path to right input video|no default|
|-l|path to left input video|no default|
|-v|define verbosity, False/True|False|
|-n|number of calibration images that should be acquired|20|
|-c|checkerboard pattern (rows,columns)|9 7|
|-o|output path|current dir|

<img
src="https://github.com/EduSampaio/OctoFishProject/blob/master/checkerboard_positions.png"
width="800">

Image 1.: Sample verbose output showing estimated 3D positions of
checkerboard corners of five calibration frames.

## Example

```
python stereo_calib.py -l 20181008-PM1-C1-calib-SYNCHED.mp4 -r 20181008-PM1-C2-calib-SNYCHED.mp4 -n 5 -c 8 6
```

If this helps, you have comments about the software or you have further
requests please feel free to email me [fritz.a.francisco@gmail.com].
