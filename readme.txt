VJCMS - stands for Viola-Jones and CAMSHIFT

1) Just build this file VJCMS and the output will be displayed
2) Ensure that you have the python plugin (opencv_ffmpeg2410_64.dll in my case) for playing the .mp4 files.
3) If you want to use to code to track faces in a different video then just add that in the videos folder and
change the file name in cv2.VideoCapture() which is on line 16
4) Tweak the values of TRACK, SKIP and RATIO according to your requirements for best output.

Thank you!