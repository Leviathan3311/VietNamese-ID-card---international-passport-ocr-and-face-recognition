import os
import string as st
from dateutil import parser
import matplotlib.image as mpimg
import cv2
from passporteye import read_mrz
import json
import easyocr
import warnings
from flask import Flask, request, jsonify

warnings.filterwarnings('ignore')
app = Flask(__name__)

# Đường dẫn đến thư mục lưu file JSON
output_dir = '/Users/leviathanvo/Documents/extracted_passport'

with open('/Users/leviathanvo/Downloads/all_country_codes.json') as f:
    country_codes = json.load(f)

def parse_date(string, iob=True):
    date = parser.parse(string, yearfirst=True).date() 
    return date.strftime('%d/%m/%Y')

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
    new_im_path = 'tmp.png'
    im_path = img_name
    # Crop image to Machine Readable Zone(MRZ)
    mrz = read_mrz(im_path, save_roi=True)

    if mrz:
        mpimg.imsave(new_im_path, mrz.aux['roi'], cmap='gray')
    
        img = cv2.imread(new_im_path)
        img = cv2.resize(img, (1110, 140))
        
        allowlist = st.ascii_letters+st.digits+'< '
        reader = easyocr.Reader(['en'])
        code = reader.readtext(img, paragraph=False, detail=0, allowlist=allowlist)
        a, b = code[0].upper(), code[1].upper()
        
        if len(a) < 44:
            a = a + '<'*(44 - len(a))
        if len(b) < 44:
                b = b + '<'*(44 - len(b))
                
        surname_names = a[5:44].split('<<', 1)
        if len(surname_names) < 2:
            surname_names += ['']
        surname, names = surname_names
        
        user_info['name'] = names.replace('<', ' ').strip().upper()
        user_info['surname'] = surname.replace('<', ' ').strip().upper()
        user_info['sex'] = get_sex(clean(b[20]))
        user_info['date_of_birth'] = parse_date(b[13:19])
        user_info['nationality'] = get_country_name(clean(b[10:13]))
        user_info['passport_type'] = clean(a[0:2])
        user_info['passport_number']  = clean(b[0:9])
        user_info['issuing_country'] = get_country_name(clean(a[2:5]))
        user_info['expiration_date'] = parse_date(b[21:27])
        user_info['personal_number'] = clean(b[28:42])
        
    else:
        return None
    
    os.remove(new_im_path)
    
    return user_info

@app.route('/upload-passports', methods=['POST'])
def upload_passports():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files part'})

    files = request.files.getlist('files[]')
    extracted_info_list = []

    for file in files:
        if file.filename == '':
            return jsonify({'error': 'One or more files have no selected file'})

        if file:
            # Lưu ảnh tạm thời
            file_path = 'temp_passport.jpg'
            file.save(file_path)

            # Trích xuất thông tin từ ảnh
            extracted_info = get_data(file_path)

            # Xóa ảnh tạm sau khi trích xuất
            os.remove(file_path)

            if extracted_info:
                extracted_info_list.append(extracted_info)
                # Tạo và lưu file JSON
                output_file_path = os.path.join(output_dir, f"{file.filename.split('.')[0]}.json")
                with open(output_file_path, 'w') as f:
                    json.dump(extracted_info, f, indent=4)
            else:
                extracted_info_list.append({'error': f'Cannot extract data from {file.filename}'})

    # Trả về thông tin trích xuất
    return jsonify({'extracted_info_list': extracted_info_list})

if __name__ == '__main__':
    app.run(debug=True)
