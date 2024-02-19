import cv2
import numpy as np
import face_recognition as face_rec
import os

class FaceRecognition:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

    #looks at the pictures on the folder to see the diffrent images
    def load_encoding_images(self, folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                image_path = os.path.join(folder_path, file_name)
                img = face_rec.load_image_file(image_path)
                img_encodings = face_rec.face_encodings(img, num_jitters=10)
                for encoding in img_encodings:
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(os.path.splitext(file_name)[0])

    #adjust the size of all the photos
    def resize(self, img, size):
        width = int(img.shape[1] * size)
        height = int(img.shape[0] * size)
        dimension = (width, height)
        return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)

    # cheskes if 2 faces look similar by comparing their features
    def compare_faces(self, encode1, encode2):
        results = face_rec.face_distance(encode1, encode2) < 0.5  # Adjust the threshold as needed
        return results

    #the program looks at two pictures of faces. First, it finds the faces in the pictures. Then, it tries to recognize those faces by comparing them to the ones it knows. Finally, it tells us if it thinks the faces are the same or not.
    def run_face_recognition(self, main_image_path, test_image_path):
        main_img = face_rec.load_image_file(main_image_path)
        main_img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
        main_img = self.resize(main_img, 0.50)

        test_img = face_rec.load_image_file(test_image_path)
        test_img = self.resize(test_img, 0.50)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        face_locations_main = face_rec.face_locations(main_img)
        if not face_locations_main:
            print("No face found in the main image.")
            return

        face_location_main = face_locations_main[0]
        encode_main = face_rec.face_encodings(main_img, known_face_locations=[face_location_main])[0]
        cv2.rectangle(main_img, (face_location_main[3], face_location_main[0]),
                      (face_location_main[1], face_location_main[2]), (255, 0, 255), 3)

        face_locations_test = face_rec.face_locations(test_img)
        if not face_locations_test:
            print("No face found in the test image.")
            return

        face_location_test = face_locations_test[0]
        encode_test = face_rec.face_encodings(test_img, known_face_locations=[face_location_test])[0]
        cv2.rectangle(test_img, (face_location_test[3], face_location_test[0]),
                      (face_location_test[1], face_location_test[2]), (255, 0, 255), 3)

        results = self.compare_faces([encode_main], encode_test)
        print(results)
        cv2.putText(test_img, f'{results}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('main_img', main_img)
        cv2.imshow('test_img', test_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage
fr = FaceRecognition()
fr.load_encoding_images('sample_images')  # Replace 'sample_images' with your folder path
fr.run_face_recognition('sample_images/joshik.jpg', 'sample_images/elonmusk.jpg')  # Replace with your image paths
