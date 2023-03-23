import cv2
from people_count import PersonCounter


config_path = "yolov4.cfg"
weights_path = "yolov4.weights"

class_path = "coco.names"
video_path = "video.mp4"

counter = PersonCounter(config_path, weights_path, class_path)

#Video Kayıt
cap = cv2.VideoCapture(video_path)
output_file = "output.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# VideoWriter nesnesi oluşturun ve ayarlarını yapın
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

while True:
    # Kameradan bir kare alın
    ret, frame = cap.read()

    frame = counter.count_people(frame)

    # Ekrana çıktı ver
    cv2.imshow("frame", frame)

    out.write(frame)

    # Çıkış için "q" tuşuna basın
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()