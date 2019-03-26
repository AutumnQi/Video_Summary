import cv2
import os

video_path = 'test_output_Obj_mp4'
index = 0

for folder in os.listdir(video_path):
    for file in os.listdir(video_path + '/' + folder):
        filename = os.path.split(file)[1]
        if filename[0] == 'S':
            length = filename.__len__()
            result = video_path + '/' + folder + '/' + filename[:length-4] + '.mp4'
            cap = cv2.VideoCapture(video_path + '/' + folder + '/' + filename)
            shape = (int(cap.get(3)), int(cap.get(4)))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            videowriter = cv2.VideoWriter(result, fourcc, fps, shape)
            span = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if cap.isOpened():
                for i in range(span):
                    ret, frame = cap.read()
                    # if ret == None:
                    #     break
                    videowriter.write(frame)
            cap.release()
            videowriter.release()
            cv2.destroyAllWindows()
            os.remove(video_path + '/' + folder + '/' + filename)
            print(str(index) + 'Finished')
            index = index + 1