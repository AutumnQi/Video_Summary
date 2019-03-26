import os

l = []

object_folder = './video/'
finish_folder = './test_output_Obj'

for floder in os.listdir(finish_folder):
    l.append(floder)

for video_name in l:
    os.remove(object_folder + video_name + '.mp4')

