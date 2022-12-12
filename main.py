import mss
import numpy as np
import cv2
import time
import keyboard
import torch
import pyautogui
import ctypes
import pynput

SendInput = ctypes.windll.user32.SendInput
PUL = ctypes.POINTER(ctypes.c_ulong)

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


def set_pos(pos):
    x, y = pos
    x = 1 + int(x * 65536./screenWidth)
    y = 1 + int(y * 65536./screenHeight)
    extra = ctypes.c_ulong(0)
    ii_ = pynput._util.win32.INPUT_union()
    ii_.mi = pynput._util.win32.MOUSEINPUT(x, y, 0, (0x0001 | 0x8000), 0, ctypes.cast(ctypes.pointer(extra), ctypes.c_void_p))
    command=pynput._util.win32.INPUT(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))


model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/cancering69/PycharmProjects/SeniorProject/yolov5/runs/train/exp/weights/best.pt')

screenWidth = 1920
screenHeight = 1080
fovSize = 320
fovHeight = 96
fovWidth = 96

with mss.mss() as sct:
    dimensions = sct.monitors[1]
    SQUARE_SIZE = 600

    monitor = {"top": int(screenHeight/2-fovSize/2), "left": int(screenWidth/2-fovSize/2), "width": fovSize, "height": fovSize}

fovHeight = 96
fovWidth = 96

while True:
    t = time.time()

    img = np.array(sct.grab(monitor))
    results = model(img)

    rl = results.xyxy[0].tolist()

    if len(rl) > 0:
        if rl[0][4] > .35:
            if rl[0][5] == 0:

                x1 = float(rl[0][0])
                x2 = float(rl[0][2])
                y1 = float(rl[0][1])
                y2 = float(rl[0][3])

                centerX = int((x1 + x2)/2) + screenWidth/2 - fovSize/2
                centerY = int((y1 + y2)/2) + screenHeight/2 - fovSize/2
                pos = (centerX, centerY)

                set_pos(pos)

    cv2.imshow('Definitely Not an Aimbot', np.squeeze(results.render()))

    cv2.waitKey(1)

    if keyboard.is_pressed('q'):
        break

cv2.destroyAllWindows()

