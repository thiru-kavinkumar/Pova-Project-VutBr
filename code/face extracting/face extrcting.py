
import os
import cv2
from tqdm import tqdm

src_path = "./data_perso/full" 
target_path = "./data_perso/cutted" 
dir_lst = []

face_classifier = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml")

print("Code start")

# Iterate data folder to find directory = image class
for path in os.listdir(src_path):
    # check if current path is a file
    if os.path.isdir(os.path.join(src_path, path)):
        dir_lst.append(path)

for dir_path in dir_lst:

    # if folder doesn't exit => creat it
    if not os.path.isdir(os.path.join(target_path, dir_path)):
        os.mkdir(os.path.join(target_path, dir_path))

    # iterate src folder to find image
    print(f"{dir_path} start")
    img_count = 0
    for img_path in tqdm(os.listdir(os.path.join(src_path, dir_path))):
        path = os.path.join(src_path, dir_path, img_path)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # Detect face from img
        scale_factor = 1.3
        neighbor = 5
        faces = face_classifier.detectMultiScale(img, scale_factor, neighbor)
        # Check if face exist
        if faces != ():
            # crop face from srource img
            for (x, y, w, h) in faces:
                cropped_face = img[y:y+h, x:x+w]
                cropped_face = cv2.resize(cropped_face, (200, 200))

            # Save cropped face
            cropped_img__path = os.path.join(
                target_path, dir_path, f"{img_count}.jpeg")
            cv2.imwrite(cropped_img__path, cropped_face)
            img_count += 1
    print(f"{dir_path} finish : {img_count} faces")
