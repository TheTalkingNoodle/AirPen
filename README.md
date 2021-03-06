# AirPen

# ====== Description ======

Convert any object into a pen to digitize your writing or doodles. Also converts text to speech using reference material from https://github.com/githubharald/SimpleHTR .


# ====== Dependencies ======
Python 2.x

NumPy

OpenCV

tensorflow

gtts

Matplotlib

PIL

# ====== Steps ======
Download/Clone this repository in any folder and run.

$ python main.py

a. Show any object with the following color in front of the webcam to detect object. The color can be changed by editing HSV values in main.py.

![alt text](https://raw.githubusercontent.com/TheTalkingNoodle/AirPen/master/pink.jpg)

b. Once the object is detected, a whiteboard will be shown with a pen cursor controlled by the object.

Options:
1. Press 's' to start tracking object to work as pen.
2. Press 'w' to start writing or doodling (pen shows GREEN pointer on screen)
3. Press 'e' to stop writing or doodling (pen shows RED pointer on screen)
4. Press 'r' to reset the object tracker (in case of mistracking). Press 's' again to start tracking again.
4. Press 'a' to convert writing to speech.

Sample Image:
![alt text](https://raw.githubusercontent.com/TheTalkingNoodle/AirPen/master/sample.png)

Note: If you want to use a custom colored object, run 'Color pallete.py' to adjust slider values in order to get the appropriate HSV values for your object.
