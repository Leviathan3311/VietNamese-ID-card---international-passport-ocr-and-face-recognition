# face verification with the VGGFace2 model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from utils import preprocess_input
from vggface import VGGFace
import sys
import cv2






# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
    img = plt.imread(filename)
    detector = MTCNN()
    results = detector.detect_faces(img)
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = img[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array


# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filenames):
    # extract faces
    faces = [extract_face(f) for f in filenames]
    # convert into an array of samples
    samples = np.asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)

    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    pred = model.predict(samples)
    return pred


# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.5):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)

    print('*****************************************************************')
    print('Threshold for the face similarity score is 0.5')

    if score <= thresh:
        print('Face is a Match with score of %.3f' % score)
    else:
        print('Face is not a Match with score of %.3f' % score)

    print('********************************************************************')


def capture_and_extract_face():
    # Tạo một cửa sổ mới
    cv2.namedWindow("Camera")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Hiển thị camera
        cv2.imshow('Camera', frame)

        # Thêm nút bấm "Chụp ảnh" và "Tắt camera"
        cv2.putText(frame, "Press 's' to capture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Camera', frame)

        # Chờ phím 's' được nhấn để chụp ảnh
        if cv2.waitKey(1) & 0xFF == ord('s'):
            # Lưu ảnh được chụp
            cv2.imwrite('captured_image.jpg', frame)
            # Trích xuất khuôn mặt từ ảnh đã chụp
            face_array = extract_face('captured_image.jpg')
            break

        # Chờ phím 'q' được nhấn để tắt camera
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

    return face_array

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

    return face_array

def main():
    # Chụp và trích xuất khuôn mặt từ ảnh đã chụp
    face_array = capture_and_extract_face()

    # Lấy embedding của khuôn mặt từ ảnh đã chụp
    embeddings = get_embeddings(['captured_image.jpg', sys.argv[1]])

    # So sánh embedding của khuôn mặt từ ảnh đã chụp với embedding của ảnh đầu vào từ dòng lệnh
    is_match(embeddings[0], embeddings[1])


if __name__ == '__main__':
    main()
