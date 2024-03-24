const express = require('express');
const multer = require('multer');
const { exec } = require('child_process');

const path = require('path');

const app = express();
const port = 3000;

const fs = require('fs');

function setFilePermissions(filePath, permissions) {
    fs.chmodSync(filePath, permissions);
}

// Sử dụng hàm setFilePermissions để cấp quyền truy cập cho tệp cụ thể
const filePath = 'main.py';
const permissions = 0o777; // Ví dụ: quyền truy cập rộng nhất (read, write, execute cho mọi người)
setFilePermissions(filePath, permissions);

// Set up directories for storing images and JSON data
const uploadPath = path.join(__dirname, 'uploads');
const extractedInfoPath = path.join(__dirname, 'extracted_info');

// Create directories if they don't exist
fs.promises.mkdir(uploadPath, { recursive: true })
    .catch(err => console.error('Error creating upload directory:', err));
fs.promises.mkdir(extractedInfoPath, { recursive: true })
    .catch(err => console.error('Error creating extracted info directory:', err));

// Set up multer middleware for handling file uploads
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, uploadPath);
    },
    filename: function (req, file, cb) {
        cb(null, file.originalname); // Keep the original filename
    }
});
const upload = multer({ storage: storage });

// Route to serve HTML file containing upload form and camera button
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'views', 'index.html'));
});

// Route to handle image upload and perform OCR
app.post('/upload', upload.single('image'), (req, res) => {
  const imagePath = req.file.path; // Get the path of the uploaded image

  // Determine the OCR function to use based on the request body
  const ocrFunction = req.body.type === 'id' ? 'main.py' : 'passport_ocr_code.py';

  // Execute the Python script to perform OCR and save the result
  const execPathOCR = path.join(__dirname, ocrFunction);
  exec(`python "${execPathOCR}" "${imagePath}"`, (errorOCR, stdoutOCR, stderrOCR) => {
      if (errorOCR) {
          console.error(`Error during OCR: ${errorOCR}`);
          res.status(500).send('Error during OCR process');
          return;
      }

      console.log(`OCR stdout: ${stdoutOCR}`);
      console.error(`OCR stderr: ${stderrOCR}`);

      // After OCR completion, read the corresponding JSON file and send its data as response
      const fileName = req.file.originalname.split('.jpg')[0];
      const jsonFilePath = path.join('/Users/leviathanvo/Documents/cccd_passport_ocr_api/extracted_info', `${fileName}.json`);
      fs.promises.readFile(jsonFilePath)
          .then(data => {
              const jsonData = JSON.parse(data);

              // Execute the Python script to perform identity verification
              const execPathVerification = path.join(__dirname, 'Face-Similarity/FaceSimilarity.py');
              exec(`python "${execPathVerification}" "${imagePath}"`, (errorVerification, stdoutVerification, stderrVerification) => {
                  if (errorVerification) {
                      console.error(`Error during identity verification: ${errorVerification}`);
                      res.status(500).send('Error during identity verification process');
                      return;
                  }

                  console.log(`Verification stdout: ${stdoutVerification}`);
                  console.error(`Verification stderr: ${stderrVerification}`);

                  // Send OCR result and verification result as part of response
                  // Send OCR result and verification result as part of response
                res.send({
                    ocrResult: createTableFromJSON(jsonFilePath), // Send OCR result
                    verificationResult: stdoutVerification // Send verification result
                });
              });
          })
          .catch(err => {
              console.error(`Error reading JSON file: ${err}`);
              res.status(500).send('Internal Server Error');
          });
  });
});

function createTableFromJSON(jsonFilePath) {
    // Đọc dữ liệu từ file JSON
    const jsonData = fs.readFileSync(jsonFilePath);
    const data = JSON.parse(jsonData);

    // Kiểm tra xem dữ liệu có phải là mảng không
    if (!Array.isArray(data)) {
        console.error('Dữ liệu không phải là một mảng.');
        return '';
    }

    // Tạo bảng HTML từ dữ liệu JSON
    let table = '<table border="1">';
    table += '<tr>';
    for (let key in data[0]) {
        table += `<th>${key}</th>`;
    }
    table += '</tr>';
    data.forEach(item => {
        table += '<tr>';
        for (let key in item) {
            table += `<td>${item[key]}</td>`;
        }
        table += '</tr>';
    });
    table += '</table>';

    return table;
}



// Start the server
app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
