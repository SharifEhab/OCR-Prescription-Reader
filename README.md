# AI-Powered OCR Prescription Reader

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Tesseract OCR](https://img.shields.io/badge/Tesseract-OCR-blue?style=for-the-badge) ![AI Integration](https://img.shields.io/badge/AI-Integration-green?style=for-the-badge) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge)

The **OCR Prescription Reader** is a hybrid system that combines advanced image processing, Tesseract OCR for text extraction, and AI-powered validation to enhance accuracy. This project automates the process of extracting, validating, and structuring prescription data, reducing errors and improving healthcare workflows.

![image](https://github.com/user-attachments/assets/009294b0-0aa7-439f-98c4-a99acae23c08)

---

## Key Features

1. **Image Preprocessing**:
   - Noise reduction, adaptive thresholding, and morphological operations for improved OCR accuracy.
2. **Text Extraction**:
   - Utilizes Tesseract OCR to extract text from prescription images.
3. **AI Integration**:
   - Validates and refines extracted text using the **Gemini AI Model**, ensuring consistency and accuracy.
4. **Drug Interaction Check**:
   - RxNorm
5. **PDF Output**:
   - Generates a structured PDF report with extracted and validated data.
![image](https://github.com/user-attachments/assets/9237016d-3431-4f8d-b9b7-c8ad6ee495e2)

---

## System Workflow

### 1. Image Preprocessing
- **Steps**:
  1. Convert image to grayscale for simplicity.
  2. Apply denoising using Non-Local Means filtering.
  3. Perform adaptive thresholding to binarize the image.
  4. Invert the binary image and apply dilation for contour detection.
- **Purpose**: Enhances text clarity and prepares the image for OCR.

### 2. Optical Character Recognition (OCR)
- Uses Tesseract OCR to detect and extract text regions from preprocessed images.
- Bounding boxes are generated for potential text areas, which are cropped, resized, and analyzed.

### 3. AI Validation
- Extracted text is passed to the Gemini AI Model for validation and correction.
- **Gemini Enhancements**:
  - Corrects abbreviations and errors in medication names.
  - Standardizes dosage units and frequencies (e.g., "QD" → "once daily").
  - Structures the data into a clear and consistent format.

### 4. Output Generation
- Results are saved as:
  - A **JSON file** for programmatic use.
  - A **PDF report** for easy sharing and documentation.

---

## Example Workflow

1. **Image Input**:
   - Upload a prescription image via the web interface.
2. **Processing**:
   - Preprocess the image and extract text using Tesseract OCR.
3. **Validation**:
   - AI validates and structures the extracted text.
4. **Output**:
   - View structured text and download the PDF report.

---

## Limitations

1. **Handwritten Text**:
   - Tesseract OCR struggles with highly variable handwriting.
   - Designed primarily for printed prescriptions.
2. **Image Quality**:
   - Poor-quality or low-resolution images may impact accuracy.

---

## Future Directions

1. **Enhanced Models**:
   - Replace Tesseract with advanced deep learning-based OCR models like CRNN or Vision Transformers for better handwriting recognition.
2. **Edge Deployment**:
   - Optimize for mobile and IoT devices for on-the-go prescription analysis.
3. **Multilingual Support**:
   - Extend support to non-English prescriptions and international drug standards.

---

## Screenshots

### **Image Preprocessing**
![preprocessed_prescription](https://github.com/user-attachments/assets/421096d9-0244-4dff-bb7c-abe00519d33e)
![image](https://github.com/user-attachments/assets/804bf1e8-8ab7-47e1-940e-0082b413c84d)
![image](https://github.com/user-attachments/assets/a98b12fd-e4a5-408b-94ad-a05673b9e4fe)
![image](https://github.com/user-attachments/assets/588a9676-261f-48d3-99ac-33b76571dc7b)
![image](https://github.com/user-attachments/assets/58d97e40-ddc4-4a3e-8e4e-81e8936a662c)

### **Extracted and Validated Text**
![image](https://github.com/user-attachments/assets/3b4920f1-3274-4215-bfcd-1f943fd0a950)
![image](https://github.com/user-attachments/assets/9f39b052-f892-414e-8297-8214569ea0e5)

---

## Requirements

1. **Environment**:
   - Python 3.8+
   - Tesseract OCR installed locally.
2. **Dependencies**:
   - `numpy`, `opencv-python`, `pytesseract`, `requests`, `fpdf`, `dotenv`, and `google.generativeai`.
  
## Authors

| Name | GitHub | LinkedIn |
| ---- | ------ | -------- |
| Abdulmonem Elsherif | [@AbdulmonemElsherif](https://github.com/AbdulmonemElsherif) | [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abdulmonem-elsherif?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app) |
| Sharif Ehab | [@Sharif_Ehab](https://github.com/SharifEhab) | [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sharif-elmasry-b167a3252/) |

