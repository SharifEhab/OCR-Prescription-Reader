import cv2
import pytesseract
import re
from PIL import Image, ImageEnhance
import numpy as np
# import openai  # Remove this import
import google.generativeai as genai  # Update this import

class PrescriptionOCR:
    def __init__(self):
        # Initialize Tesseract path
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # for Windows

        # Updated units and frequencies
        self.units = ['mg', 'ml', 'mcg', 'g', 'tab', 'tabs', 'tablet', 'tablets']
        self.frequencies = [
            'bid', 'tid', 'qid', 'qd', 'prn', 'od', 'qod',
            'daily', 'twice daily', 'three times a day',
            'q4h', 'q6h', 'q8h', 'q12h',
            'ac', 'pc'
        ]

        # Initialize Gemini with your valid API key
        genai.configure(api_key='AIzaSyBOu4klXu5CKu5hSahOIeZT2s8a4j9_ol8')  # Replace with your real API key
        self.model = genai.GenerativeModel('gemini-pro')

    def correct_ocr_errors(self, text):
        # Common OCR misreads
        corrections = {
            '0': ['O', 'o'],
            '1': ['I', 'l', '|'],
            '2': ['Z'],
            '5': ['S', 's'],
            '8': ['B'],
            'mg': ['mg', 'mg.', 'mg,', 'mq', 'mq.'],
            'ml': ['ml', 'ml.', 'ml,', 'mi', 'mi.'],
            'tab': ['tab', 'tob', 'tab.', 't4b', 't4.b'],
            'tabs': ['tabs', 'tobs', 'tabs.'],
            'bid': ['bid', 'b1d', 'bld', 'bid.'],
            'tid': ['tid', 't1d', 'tld', 'tid.', 'TH'],
            'qid': ['qid', 'q1d', 'qld', 'qid.'],
            'prn': ['prn', 'prm', 'prn.'],
            'qd': ['qd', 'q.d', 'QD'],
            'qod': ['qod', 'q0d', 'qod.']
        }

        for correct, incorrect_list in corrections.items():
            for incorrect in incorrect_list:
                text = re.sub(r'\b' + re.escape(incorrect) + r'\b', correct, text, flags=re.IGNORECASE)
        return text

    def preprocess_image(self, image_path):
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Could not read the image. Please check the image path.")
    
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        # Apply Gaussian Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
        # Apply binary inverse thresholding
        thresh = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
        # Apply dilation to connect text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=1)
    
        # Find contours in the dilated image
        contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        print(f"Contours detected: {len(contours)}")
    
        # Draw bounding boxes on the original image
        image_with_boxes = image.copy()
    
        # Draw bounding boxes around contours
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter out small contours to reduce noise
            if w > 30 and h > 15:
                cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
        # Save the image with bounding boxes
        cv2.imwrite('preprocessed_with_boxes.png', image_with_boxes)
    
        # Return the thresholded image for OCR
        return thresh

    def extract_medication_info(self, text):
        medications = []

        # Correct common OCR errors in the text
        text = self.correct_ocr_errors(text)

        # Split text into lines and remove empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # Regular expressions for medication information
        dosage_pattern = r'(\d+(?:\.\d+)?)\s*(mg|ml|mcg|g)'
        tab_pattern = r'(\d+)\s*(tab|tabs|tablet|tablets)'
        frequency_pattern = r'\b(' + '|'.join(self.frequencies) + r')\b'

        for line in lines:
            # Skip header and footer lines
            if any(header in line.lower() for header in ['name:', 'address:', 'date:', 'age:', 'rx', 'medical centre', 'signature', 'refill', 'label', 'prescription']):
                continue

            # Convert to lowercase for matching
            line_lower = line.lower()

            # Extract dosage
            dosages = re.findall(dosage_pattern, line_lower)
            tab_doses = re.findall(tab_pattern, line_lower)

            # Combine dosage information
            full_dosage = []
            if dosages:
                full_dosage.extend(f"{d[0]}{d[1]}" for d in dosages)
            if tab_doses:
                full_dosage.extend(f"{d[0]} {d[1]}" for d in tab_doses)

            # Extract frequency
            frequencies = re.findall(frequency_pattern, line_lower)

            # Extract medication name (words before dosage)
            parts = line.split()
            med_name = ''
            for i, word in enumerate(parts):
                if re.match(dosage_pattern, word.lower()) or re.match(tab_pattern, word.lower()):
                    med_name = ' '.join(parts[:i])
                    break
            if not med_name and parts:
                med_name = parts[0]

            if med_name and (full_dosage or frequencies):
                medication_info = {
                    'medication': med_name.capitalize(),
                    'dosage': ' + '.join(full_dosage) if full_dosage else 'Not specified',
                    'frequency': frequencies[0] if frequencies else 'Not specified',
                    'original_line': line
                }
                medications.append(medication_info)

        return medications

    def extract_text(self, preprocessed_image):
        try:
            # OCR configuration
            custom_config = r'--oem 3 --psm 6'

            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(preprocessed_image)

            # Enhance image contrast
            enhancer = ImageEnhance.Contrast(pil_image)
            enhanced_image = enhancer.enhance(1.5)

            # Perform OCR
            text = pytesseract.image_to_string(
                enhanced_image,
                config=custom_config,
                lang='eng'
            )

            # Clean the text
            text = re.sub(r'[^\w\s.-]', ' ', text)
            text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())

            print("Raw OCR Output:")
            print("-" * 50)
            print(text)
            print("-" * 50)
            return text
        except Exception as e:
            print(f"OCR Error: {str(e)}")
            return ""

    def validate_medications(self, medications):
        """
        Validate extracted medications using Gemini to check for errors and inconsistencies
        """
        try:
            # Construct a more explicit and safer prompt
            prompt = """As a medical prescription validator, please review the following extracted prescription data 
            for technical accuracy of names, dosages, and frequencies only. Do not provide medical advice or judgments.
            Please only check for formatting and obvious typographical errors:\n\n"""
            
            for med in medications:
                prompt += f"- {med['medication']}, {med['dosage']}, {med['frequency']}\n"

            # Get validation from Gemini
            response = self.model.generate_content(prompt)
            
            # Handle the response properly
            if response.candidates and response.candidates[0].content:
                validated_response = response.candidates[0].content.parts[0].text
            else:
                validated_response = "Validation could not be performed. Please verify prescription manually."
            
            print("\nAI Validation Results:")
            print("-" * 50)
            print(validated_response)
            print("-" * 50)
    
            return validated_response
    
        except Exception as e:
            print(f"Validation Error: {str(e)}")
            return None

    def read_prescription(self, image_path):
        try:
            # Preprocess the image
            print(f"Processing image: {image_path}")
            preprocessed_image = self.preprocess_image(image_path)

            # Save preprocessed image for debugging
            cv2.imwrite('preprocessed_prescription.png', preprocessed_image)
            print("Saved preprocessed image as 'preprocessed_prescription.png'")

            # Extract text from the preprocessed image
            text = self.extract_text(preprocessed_image)

            if not text.strip():
                return "No text was extracted from the image. Please check image quality."

            # Extract medication information
            medications = self.extract_medication_info(text)

            if not medications:
                return "No medication information could be extracted. Raw text was:\n" + text

            # Add validation step
            validated_results = self.validate_medications(medications)
            
            return {
                'extracted_medications': medications,
                'validation_results': validated_results
            }

        except Exception as e:
            return f"Error processing prescription: {str(e)}"

def main():
    # Create an instance of PrescriptionOCR
    ocr_reader = PrescriptionOCR()

    # Example usage
    image_path = "prescription.jpg"
    results = ocr_reader.read_prescription(image_path)

    # Print results
    if isinstance(results, dict):
        print("\nExtracted Prescription Details:")
        for med in results['extracted_medications']:
            print("\nMedication:", med['medication'])
            print("Dosage:", med['dosage'])
            print("Frequency:", med['frequency'])
        
        print("\nAI Validation Results:")
        print(results['validation_results'])
    else:
        print(results)

if __name__ == "__main__":
    main()