import cv2
import numpy as np


class PersonCounter:
    def __init__(self, config_path, weights_path, class_path):
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.classes = []
        with open(class_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.detected_people = {}
        self.next_person_id = 1  # Başlangıç ID'si
        self.person_id = 0


    def count_people(self, frame):
        # Giriş karesini YOLO için işleme sokun
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)

        # YOLO ağırlıkları kullanarak tespitler yapın
        self.net.setInput(blob)
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        outs = self.net.forward(output_layers)

        # Tespitlerin sınıflandırılması ve sayım yapılması
        conf_threshold = 0.1
        nms_threshold = 0.4
        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold and class_id == 0: #İnsan
                    # Tespitin koordinatlarını ve boyutunu hesaplayın
                    box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (center_x, center_y, width, height) = box.astype('int')

                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))

                    # Tespitin sınıfı, güven değeri ve koordinatlarını kaydedin
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, int(width), int(height)])


        # Non-maximum suppression (NMS) uygulayın
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # Eşleşen bir kişi var mı kontrol edin
        if len(indices) > 0:
            person_found = [False] * len(self.detected_people)
            for i in indices.flatten():
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]

                # Eşleşen bir kişi var mı kontrol edin
                for j, (person_id, (px, py, pw, ph)) in enumerate(self.detected_people.items()):
                    center_x = x + w / 2
                    center_y = y + h / 2
                    if center_x > px and center_x < px + pw and center_y > py and center_y < py + ph:
                        self.detected_people[person_id] = (x, y, w, h)
                        person_found[j] = True
                        person_id_found = person_id  # Eşleşen bir ID bulundu
                        break

                # Yeni bir kişi tespit edildiyse, sayacı arttırın
                if not any(person_found):
                    self.person_id += 1
                    self.detected_people[self.person_id] = (x, y, w, h)
                    person_found.append(True)
                    person_id_found = self.person_id  # Yeni bir ID oluşturuldu


            # Yeni bir kareye geçmeden önce, tespit edilen insanları kaydetmek için detected_people değişkenini güncelleyin
            self.detected_people = {k: v for j, (k, v) in enumerate(self.detected_people.items()) if person_found[j]}
            #detected_people = {k: v for k, v in detected_people.items() if person_found[k]}
            person_found = [False] * len(self.detected_people)

        
        # Ekrana çıktı ver
        for person_id, (x, y, w, h) in self.detected_people.items():
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {person_id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            last_id = max(self.detected_people.keys()) if self.detected_people else 0
            cv2.putText(frame, f"Last ID: {last_id}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        return frame

