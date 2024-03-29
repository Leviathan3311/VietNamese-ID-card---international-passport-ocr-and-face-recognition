import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import json
import sys
from ultralytics import YOLO
from PIL import Image
from mtcnn import MTCNN
import cv2
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import logging
import keras
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
keras.utils.disable_interactive_logging()

logging.basicConfig(level=logging.DEBUG)


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

names_index = {9: 'id', 12: 'full_name', 4: 'birth', 8: 'gender', 13: 'nationality', 14: 'place_of_origin',
               15: 'place_of_residences', 7: 'date_of_expiry', 10: 'issue_date', 11: 'issue_place'}
imgWidth = 640
imgHeight = 480

# YOLO
sources_model = '/Users/leviathanvo/Documents/cccd_passport_ocr_api/cccdYolov8.pt'
model = YOLO(sources_model)

# VietOcr
config = Cfg.load_config_from_name('vgg_seq2seq')
config['device'] = 'cpu'
config['predictor']['beamsearch'] = False
detector = Predictor(config)


def ocr(crop_img):
    text = detector.predict(crop_img)
    return text


def detect_and_crop_faces(image_path, output_dir, file_name, padding=40):
    image = cv2.imread(image_path)
    detector = MTCNN()
    faces = detector.detect_faces(image)

    for i, face in enumerate(faces):
        x, y, width, height = face['box']
        x -= padding
        y -= padding
        width += padding * 2
        height += padding * 2

        x = max(0, x)
        y = max(0, y)
        width = min(width, image.shape[1] - x)
        height = min(height, image.shape[0] - y)

        cropped_face = image[y:y + height, x:x + width]

        output_path = os.path.join(output_dir, file_name + '.jpg')

        cv2.imwrite(output_path, cropped_face)


def get_text(img_path):
    img = Image.open(img_path)
    results = model.predict(source=img_path)
    dic = {'id': [], 'full_name': [], 'birth': [], 'gender': [], 'nationality': [], 'place_of_origin': [],
           'place_of_residences': [], 'date_of_expiry': [], 'issue_date': [], 'issue_place': []}

    for box in results[0].boxes:
        if int(box.cls[0]) in names_index:
            name = names_index[int(box.cls[0])]
            dic[name].append(box.xyxy[0].cpu().numpy().astype(int))

    res = {'id': '', 'full_name': '', 'birth': '', 'gender': '', 'nationality': '', 'place_of_origin': '',
           'place_of_residences': '', 'date_of_expiry': '', 'issue_date': '', 'issue_place': ''}

    for key in dic:
        tmp = ''
        flag = 0
        for value in dic[key]:
            crop_img = img.crop(value)
            if key == 'place_of_residences' and flag <= 1:
                res[key] = ocr(crop_img) + tmp
                tmp = ',' + ' ' + ocr(img.crop(value))
                flag += 1
            elif flag > 1:
                continue
            else:
                res[key] = tmp + ocr(crop_img)

    return res


def save_json(data, output_dir, filename):
    json_file_path = os.path.join(output_dir, filename + '.json')
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)



def get_info(img_path, output_dir):
    detect_and_crop_faces(img_path, output_dir, 'image')
    res = get_text(img_path)

    return res




def main(image_path, file_name):
    logging.debug(f"Received image path: {image_path}")
    result = []
    folder_link = os.path.join('extracted_info', file_name)
    new_output_dir = folder_link
    i = 2
    while os.path.exists(new_output_dir):
        if i == 2:
            new_output_dir = f"{folder_link}_{i}" 
        else:
            new_output_dir = f"{new_output_dir[:-2]}_{i}"  # Lấy phần tên trước "_" và thêm số tiếp theo         
        i +=1
    os.makedirs(new_output_dir)
    if os.path.isdir(image_path):
            for filename in os.listdir(image_path):
                # Kiểm tra xem tệp có phải là một ảnh hay không
                if filename.endswith((".jpg", "png", ".JPEG")):
                    # Đường dẫn đầy đủ đến tệp ảnh
                    path = os.path.join(image_path, filename)
                    # Lưu kết quả
                    result.append(get_info(path, new_output_dir))
    else:
        print('không bắt được file')
        
    # Gộp các phần tử và loại bỏ các giá trị rỗng
    combined_data = {}
    for item in result:
        filtered_item = {key: value for key, value in item.items() if value.strip() != ''}
        combined_data.update(filtered_item)
    save_json(combined_data, new_output_dir, 'info')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
