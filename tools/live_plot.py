import sys

import time
from numpy import arange
import matplotlib.pyplot as plt

PLOT_WIDTH=500
plt.axis([0, PLOT_WIDTH, -90, 90])
plt.ion()

pitch = [0]
yaw = [0]
roll = [0]
pitch_graph, yaw_graph, roll_graph = plt.plot(pitch, 'r', yaw, 'g', roll, 'b')

plt.legend([pitch_graph, yaw_graph, roll_graph], ['Pitch', 'Yaw', 'Roll'])
plt.show()


while True:
    line = sys.stdin.readline()
    data = eval(line)
    if "face_0" in data:

        pitch.append(data["face_0"]["pitch"])
        pitch_graph.set_data(arange(0, len(pitch)), pitch)

        yaw.append(data["face_0"]["yaw"])
        yaw_graph.set_data(arange(0, len(yaw)), yaw)

        roll.append(data["face_0"]["roll"])
        roll_graph.set_data(arange(0, len(roll)), roll)

        plt.xlim([max(0,len(pitch) - PLOT_WIDTH), max(0,len(pitch) - PLOT_WIDTH) + PLOT_WIDTH])
        plt.draw()
    else:
        pitch.append(0)
        yaw.append(0)
        roll.append(0)

