const express = require('express');
const WebSocket = require('ws');
const { exec } = require('child_process');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const bodyParser = require('body-parser');

const app = express();
const server = app.listen(3000);
const wss = new WebSocket.Server({ server });

let currentInterval = null;

const uploadPath = path.join(__dirname, 'uploads');
const extractedInfoPath = path.join(__dirname, 'extracted_info');

app.use('/extracted_info', express.static(path.join(__dirname, 'extracted_info')));
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));
app.use(bodyParser.urlencoded({ extended: true }));

fs.promises.mkdir(uploadPath, { recursive: true })
    .catch(err => console.error('Error creating upload directory:', err));
fs.promises.mkdir(extractedInfoPath, { recursive: true })
    .catch(err => console.error('Error creating extracted info directory:', err));

const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, uploadPath);
    },
    filename: function (req, file, cb) {
        cb(null, file.originalname);
    }
});
const upload = multer({ storage: storage });

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'views', 'index.html'));
});

app.post('/upload', upload.fields([{ name: 'front', maxCount: 1 }, { name: 'back', maxCount: 1 }, { name: 'avatar', maxCount: 1 }, { name: 'passport', maxCount: 1 }, { name: 'passportAvatar', maxCount: 1 }]), (req, res) => {
    const selectedType = req.body.type;
    const fullname = req.body.fullname;

    let uploadFields, fileName2, avatarImagePath;
    if (selectedType === 'ID') {
        uploadFields = ['front', 'back'];
        avatarImagePath = req.files['avatar'][0].path
        fileName2 = req.files['avatar'][0].originalname.toLowerCase().split('.jpg')[0];
    } else if (selectedType === 'PASSPORT') {
        uploadFields = ['passport'];
        avatarImagePath = req.files['passportAvatar'][0].path
        fileName2 = req.files['passportAvatar'][0].originalname.toLowerCase().split('.jpg')[0];
    } else {
        return res.status(400).send('Invalid selection.');
    }

    const tempPath = path.join(__dirname, 'temp');

    // Tạo thư mục tạm thời nếu chưa tồn tại
    fs.mkdirSync(tempPath, { recursive: true });

    for (let field of uploadFields) {
        const file = req.files[field];
        moveFile(file[0].path, tempPath);
    }

    let maxTime = selectedType === 'ID' ? 15 : 20;
    let incrementTime = 1000;
    let progress = 0;

    if (currentInterval) {
        clearInterval(currentInterval);
        currentInterval = null;
    }

    currentInterval = setInterval(() => {
        progress = Math.min(progress + (100 / maxTime), 100);
        let roundedProgress = parseInt(progress);
        wss.clients.forEach(client => {
            client.send(JSON.stringify({ type: 'progress', data: roundedProgress }));
        });
        if (progress === 100) {
            clearInterval(currentInterval);
        }
    }, incrementTime);

    const ocrFunction = selectedType === 'ID' ? 'main.py' : 'passport_ocr_code.py';
    const execPathOCR = path.join(__dirname, ocrFunction);
    const originalPath = path.join('/Users/leviathanvo/Documents/cccd_passport_ocr_api/extracted_info', `${fullname}`)
    // Kiểm tra và sửa đổi đường dẫn nếu cần thiết
    const newPath = checkAndModifyPath(originalPath);
    exec(`python "${execPathOCR}" "${tempPath}" "${fullname}"`, (errorOCR, stdoutOCR, stderrOCR) => {
        if (errorOCR) {
            console.error(`Error during OCR: ${errorOCR}`);
            clearInterval(currentInterval);
            res.status(500).send('Error during OCR process');
            return;
        }
        const baseName = path.basename(newPath);

        const jsonFilePath = path.join(newPath, 'info.json');
        const inputImagePath = path.join(newPath,'image.jpg');
        const outputImagePath = `/extracted_info/${baseName}/image.jpg`;
        const outputAvaImagePath = `/uploads/${fileName2}.jpg`;

        fs.promises.readFile(jsonFilePath)
            .then(data => {
                const ocrType = selectedType === 'ID' ? 'ID' : 'PASSPORT';
                const tableHTML = createTableFromJSON(jsonFilePath);

                const execPathVerification = path.join(__dirname, 'Face-Similarity/FaceSimilarity.py');
                exec(`python "${execPathVerification}" "${inputImagePath}" "${avatarImagePath}"`, (errorVerification, stdoutVerification, stderrVerification) => {
                    if (errorVerification) {
                        console.error(`Error during identity verification: ${errorVerification}`);
                        currentInterval = null;
                        res.status(500).send('Error during identity verification process');
                        return;
                    }   

                    res.json({
                        ocrType : ocrType,
                        ocrResult: tableHTML,
                        verificationResult: stdoutVerification,
                        inputImage: outputImagePath,
                        avatarImage: outputAvaImagePath,
                    });
                    // Xoá thư mục tạm thời sau khi xử lý xong
                    fs.rmdirSync(tempPath, { recursive: true });
                });
            })
            .catch(err => {
                console.error(`Error reading JSON file: ${err}`);
                clearInterval(currentInterval);
                res.status(500).send('Internal Server Error');
            });
    });
});

function moveFile(filePath, destinationFolder) {
    const fileName = path.basename(filePath);
    const newPath = path.join(destinationFolder, fileName);
    fs.renameSync(filePath, newPath);
    return newPath;
}

function checkAndModifyPath(filePath) {
    let newFilePath = filePath;
    let i = 2;
    if (!fs.existsSync(filePath)) {
        return newFilePath; // Trả về đường dẫn ban đầu nếu không cần sửa đổi
    }
    while (fs.existsSync(newFilePath)) {
        const ext = path.extname(filePath);
        const dir = path.dirname(filePath);
        const fileName = path.basename(filePath, ext);
        newFilePath = path.join(dir, `${fileName}_${i}${ext}`);
        i++;
    }
    return newFilePath;
}

function createTableFromJSON(jsonFilePath) {
    let jsonData;
    try {
        const jsonDataString = fs.readFileSync(jsonFilePath, 'utf8');
        jsonData = JSON.parse(jsonDataString);
    } catch (error) {
        console.error('Error reading or parsing JSON file:', error);
        return '';
    }

    if (typeof jsonData !== 'object') {
        console.error('Data is not a JSON object.');
        return '';
    }

    let table = '<table border="1">';
    for (let key in jsonData) {
        table += `<tr><th>${key}</th><td>${jsonData[key]}</td></tr>`;
    }
    table += '</table>';

    return table;
}

console.log(`Server is running on http://localhost:3000`);
