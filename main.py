from ultralytics import YOLO
import cv2
import cvzone
import math
import face_recognition
import os


def calculate_overlap(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


# Preload known faces
known_faces_dir = "C:\\Users\\sambita\\PycharmProjects\\joy\\known_faces"
known_face_encodings = []
known_face_names = []

# Load all images from the known_faces directory
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(known_faces_dir, filename)
        known_image = face_recognition.load_image_file(image_path)
        known_face_encoding = face_recognition.face_encodings(known_image)[0]
        known_face_encodings.append(known_face_encoding)
        known_face_names.append(os.path.splitext(filename)[0])  # Use the filename without extension as the name

# Set up video capture
cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 640)  # Lower resolution
cap.set(4, 360)

# Load YOLO model
model = YOLO("ppe.pt")

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest',
              'machinery', 'vehicle']
myColor = (0, 0, 255)

frame_skip = 5  # Process every 5th frame
frame_count = 0

while True:
    success, img = cap.read()
    if not success:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    results = model(img, stream=True)
    person_boxes = []
    hardhat_boxes = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            print(currentClass)

            # Drawing rectangles for Hardhats, No Hardhats, and Persons
            if conf > 0.5 and (currentClass == 'Hardhat' or currentClass == 'NO-Hardhat' or currentClass == 'Person'):
                if currentClass == 'NO-Hardhat':
                    myColor = (0, 0, 255)
                elif currentClass == 'Hardhat':
                    myColor = (0, 255, 0)
                else:
                    myColor = (255, 0, 0)

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                   colorT=(255, 255, 255), colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

                # Append the bounding box to the respective list
                if currentClass == 'Hardhat' or currentClass == 'NO-Hardhat':
                    hardhat_boxes.append((x1, y1, x2, y2, currentClass))
                if currentClass == 'Person':
                    person_boxes.append((x1, y1, x2, y2))

    # Face recognition and drawing rectangles and text for recognized faces
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_box = (left, top, right, bottom)

        # Default color for unknown face
        face_color = (255, 0, 255)  # Pink

        # Check overlap with Person and Hardhat/No-Hardhat boxes
        for person_box in person_boxes:
            for hardhat_box in hardhat_boxes:
                iou_hardhat = calculate_overlap(person_box, hardhat_box)
                if iou_hardhat > 0.1:
                    iou = calculate_overlap(face_box, person_box)
                    if iou > 0.1:
                        if name != "Unknown":
                            face_color = (0, 255, 255) if hardhat_box[4] == "Hardhat" else (128, 0, 128)
                        else:
                            face_color = (0, 165, 255) if hardhat_box[4] == "Hardhat" else (255, 0, 255)
                        break

        cv2.rectangle(img, face_box, face_color, 2)
        cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, face_color, 2)

    cv2.imshow("Video", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()







































































































