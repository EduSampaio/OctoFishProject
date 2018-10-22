# Stereo calibration:

This script is designed to calibrate a stereocamera and save the 
required calibration metrix in a ouput (.npz) file. Further verbosity of 
the viewable output can be enabled by setting the equivalent flag (see 
below).

## Usage

In order to use the stereocalibraion script (command line interface) 
navigate to the 
containing folder and execute
```bash
python stereo_calib.py
```
followed by the additional arguments for specification:

|Argument       | Use           |Default |
|:-------------: |:-------------| :-----|
|-r|video path to right input |no default|
|-l|video path to left input|no default|
|-v|define verbosity, 0=False 1=True |default False|
|-n|number of calibration images that should be acquired|default 20|

<img 
src="https://github.com/EduSampaio/OctoFishProject/blob/master/checkerboard_positions.png" 
width="800">

Image 1.: Sample verbose output showing estimated 3D positions of 
checkerboard corners of five calibration frames.

If this helps, you have comments about the software or you have further 
requests please feel free to email me [fritz.a.francisco@gmail.com].
