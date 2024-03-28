import os
import string as st
from dateutil import parser
import matplotlib.image as mpimg
import cv2
from passporteye import read_mrz
import json
import easyocr
from PIL import Image
from paddleocr import PaddleOCR, draw_ocr
import warnings
from mtcnn import MTCNN
import numpy as np
from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor 
warnings.filterwarnings('ignore')
import sys 
import logging 
import keras
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
keras.utils.disable_interactive_logging()

logging.basicConfig(level=logging.DEBUG)


# os.environ['KMP_DUPLICATE_LIB_OK']='True'
with open('/Users/leviathanvo/Documents/cccd_passport_ocr_api/all_country_codes.json') as f:
    country_codes = json.load(f)
    
    
names_index = {0 : 'MRZ'}
imgWidth = 640 
imgHeight = 480

# YOLO 

sources_model = '/Users/leviathanvo/Documents/cccd_passport_ocr_api/passport_mrz_yoloV8.pt'
model = YOLO(sources_model) 


def parse_date(string, iob=True):
    try:
        string = string.replace('O', '0').replace('D','0')
        date = parser.parse(string, yearfirst=True).date() 
        return date.strftime('%Y/%m/%d')
    except:
        return string 


def clean(string):
    return ''.join(i for i in string if i.isalnum()).upper()


def get_country_name(country_code):
    country_name = ''
    for country in country_codes:
        if country['alpha_3'] == country_code:
            country_name = country['name']
            return country_name.upper()
    return country_code

def get_sex(code):
    if code in ['M', 'm', 'F', 'f']:
        sex = code.upper() 
    elif code == '0':
        sex = 'M'
    else:
        sex = 'F'
    return sex


def print_data(data):
    dic = {
        "name": "",
        "surname": "",
        "sex": "",
        "date_of_birth": "",
        "nationality": "",
        "passport_type": "",
        "passport_number": "",
        "issuing_country": "",
        "expiration_date": "",
        "personal_number": ""
    }

    for key in data.keys():
        info = key.replace('_', ' ').capitalize()
        dic[key] = data[key]
        # print(f'{info}\t:\t{data[key]}')
    return dic



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


def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def remove_noise(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return cv2.fastNlMeansDenoisingColored(blurred, None, 10, 10, 7, 21)

    
def scale_to_dpi(image, target_dpi):
    # Định nghĩa DPI hiện tại của ảnh (pixels per inch)
    current_dpi = 72  # DPI mặc định của OpenCV
    
    # Tính tỉ lệ cần phải thay đổi để đạt được độ phân giải mục tiêu
    scale_factor = target_dpi / current_dpi
    
    # Tính kích thước mới của ảnh dựa trên tỉ lệ
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    
    # Thực hiện chuyển đổi kích thước ảnh
    scaled_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    
    return scaled_image


def remove_trailing_K(string):
    while (string.endswith('K') and string[-2] == ' ') or (string.endswith('K') and string[-2] == 'K'):
        string = string[:-1]
        string = string.rstrip()
    return string


def get_data(img_name):
    """ Extract personal info from img_name

    Args:
        img_name (str or fp): name or path of the passport image

    Returns:
        dict: dictionary of extracted data with keys and corresponding values as follows:
                surname         : surname
                name            : name
                sex             : sex
                date_of_birth   : date of birth
                nationality     : nationality
                passport_type   : passport type
                issuing_country : issuing country
                expiration_date : expiration date
                personal_number : personal number           
    """
    
    user_info = {}
    img = Image.open(img_name)
    new_im_path = 'tmp.png'
    im_path = img_name
    results = model.predict(source=im_path, device='mps')
    box = results[0].boxes
    name = box.xyxy[0].cpu().numpy().astype(int)
    crop_img = img.crop(name)
    crop_img.save(new_im_path)
#     crop_img.show()
    img = cv2.imread(new_im_path)
#     deskew_img = deskew(img)
    cv2.imwrite(new_im_path, img)

    
    
    # Crop image to Machine Readable Zone(MRZ)
    mrz = read_mrz(new_im_path, save_roi=True)

    if mrz:
        mpimg.imsave(new_im_path, mrz.aux['roi'], cmap='gray')
 
        img = cv2.imread(new_im_path)
        img = cv2.resize(img, (1110, 140))
     
        img = remove_noise(img)
        img = cv2.convertScaleAbs(img, alpha=1.5, beta=1.3)
        img = scale_to_dpi(img, 1000)

        
        allowlist = st.ascii_letters+st.digits+'< '
        reader = easyocr.Reader(['en'])
        code = reader.readtext(img, paragraph=False, detail=0, allowlist=allowlist)

        
        if len(code) == 2:
            a, b= code[0].upper(), code[1].upper()

        else: 
            # Tạo danh sách mới chỉ chứa các phần tử có độ dài lớn hơn hoặc bằng 40
            filtered_data = [item for item in code if len(item) >= 30]

            # Sắp xếp danh sách mới theo thứ tự giảm dần độ dài của các phần tử
            sorted_filtered_data = sorted(filtered_data, key=len)

            # Select the first two strings
            a, b = sorted_filtered_data[:2]


        if len(a) < 44:
            a = a + '<'*(44 - len(a))
        if len(b) < 44:
            b = b + '<'*(44 - len(b))

        surname_names = a[5:44].split('<<', 1)
        if len(surname_names) < 2:
            surname_names += ['']
        surname, names = surname_names
        surname = surname.replace('<', ' ').strip().rstrip().upper()
        names = names.replace('4', ' ').replace('<', ' ').rstrip()
        # Tìm vị trí của ký tự khoảng trắng đầu tiên
        space_index = surname.find(' ')
        tmp = surname[space_index+1:]
        # Kiểm tra nếu có ký tự khoảng trắng trong phần sau họ
        if ' ' in surname[space_index+1:] or tmp.isalpha() and space_index != -1:
            # Nếu có ký tự khoảng trắng, giả sử rằng đó là phần tên và phần phía trước là họ
            names = surname[space_index+1:]
            surname = surname[:space_index]
     
        
        names = remove_trailing_K(names)
        user_info['name'] = names.upper()
        user_info['surname'] = surname
        user_info['sex'] = get_sex(clean(b[20]))
        user_info['date_of_birth'] = parse_date(b[13:19])
        user_info['nationality'] = get_country_name(clean(b[10:13]))
        user_info['passport_type'] = clean(a[0:2])
        user_info['passport_number']  = clean(b[0:9])
        user_info['issuing_country'] = get_country_name(clean(a[2:5]))
        user_info['expiration_date'] = parse_date(b[21:27])
        user_info['personal_number'] = clean(b[28:42])
            
            
        os.remove(new_im_path) 
        return print_data(user_info)

       
    else:
        os.remove(new_im_path)
        return print(f'Machine cannot read image {img_name}.')
    

def save_json(data, output_dir, filename):
    json_file_path = os.path.join(output_dir, filename + '.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


        
def get_info(path, output_dir):
    detect_and_crop_faces(path, output_dir, 'image')
    r = get_data(path)
    return r
        

def main(image_path, file_name):
    logging.debug(f"Received image path: {image_path}")
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
                    result = get_info(path, new_output_dir)
                    save_json(result, new_output_dir, 'info')
    else:
        print('không bắt được file')



if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
            
