import os
import re
import json
import pytesseract
import google.generativeai as genai
from getpass import getpass
import requests
import numpy as np
import cv2
from PIL import Image
from dotenv import load_dotenv
from fpdf import FPDF


class PrescriptionProcessor:
    def __init__(self, image_url):
        self.image_url = image_url
        self.image = None
        self.binary_image = None
        self.inverted_image = None
        self.dilated_image = None
        self.contours = []
        self.handwriting_regions = []
        self.merged_boxes = []
        self.filtered_boxes = []
        self.bounding_box_images = []
        self.extracted_text_data = {}
        self.gemini_response = None  # To store Gemini output

         # Load environment variables
        load_dotenv()
        self.gemini_key = os.getenv("GEMINI_API_KEY")

    def load_and_preprocess_image(self):
        # Load image from URL
        self.image = Image.open(requests.get(self.image_url, stream=True).raw)
        grayscale_image = self.image.convert('L')
        grayscale_array = np.array(grayscale_image)

        # Denoising the image
        filtered_image = cv2.fastNlMeansDenoising(grayscale_array, None, 30, 7, 21)

        # Adaptive Thresholding
        self.binary_image = cv2.adaptiveThreshold(
            filtered_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )

        # Inverting the binary image
        self.inverted_image = cv2.bitwise_not(self.binary_image)

    def apply_dilation(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Adjust kernel size
        self.dilated_image = cv2.dilate(self.inverted_image, kernel, iterations=1)

    def detect_contours(self):
        contours, _ = cv2.findContours(self.dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = contours

    def generate_bounding_boxes(self):
        for contour in self.contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            area = w * h

            # Only keep bounding boxes within a valid range
            if 0.3 < aspect_ratio < 17.0 and 100 < area < 19000:
                self.handwriting_regions.append((x, y, w, h))

    def merge_bounding_boxes(self):
        # Sort the regions by their y-coordinate for proper vertical alignment
        self.handwriting_regions.sort(key=lambda box: box[1])

        for i, (x1, y1, w1, h1) in enumerate(self.handwriting_regions):
            merged = False
            for j in range(len(self.merged_boxes)):
                x2, y2, w2, h2 = self.merged_boxes[j]
                if abs(y1 - y2) < 15:  # Adjust threshold for vertical closeness
                    self.merged_boxes[j] = (
                        min(x1, x2),
                        min(y1, y2),
                        max(x1 + w1, x2 + w2) - min(x1, x2),
                        max(y1 + h1, y2 + h2) - min(y1, y2),
                    )
                    merged = True
                    break

            if not merged:
                self.merged_boxes.append((x1, y1, w1, h1))

    def remove_overlapping_boxes(self, iou_threshold=0.01):
        def compute_iou(box1, box2):
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)

            if x_right < x_left or y_bottom < y_top:
                return 0.0  # No overlap
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            box1_area = w1 * h1
            box2_area = w2 * h2
            return intersection_area / float(box1_area + box2_area - intersection_area)

        filtered_boxes = []
        while self.merged_boxes:
            self.merged_boxes = sorted(self.merged_boxes, key=lambda b: b[2] * b[3], reverse=True)
            chosen_box = self.merged_boxes.pop(0)
            filtered_boxes.append(chosen_box)

            self.merged_boxes = [
                box
                for box in self.merged_boxes
                if compute_iou(chosen_box, box) < iou_threshold
            ]

        self.filtered_boxes = filtered_boxes

    def crop_and_resize_boxes(self, max_size=256):
        filtered_boxes_sorted = sorted(self.filtered_boxes, key=lambda box: box[1])
        for idx, (x, y, w, h) in enumerate(filtered_boxes_sorted):
            roi = self.binary_image[y:y + h, x:x + w]
            aspect_ratio = w / h

            # Set max size for resizing while maintaining aspect ratio
            if aspect_ratio > 1:
                new_width = min(w, max_size)
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = min(h, max_size)
                new_width = int(new_height * aspect_ratio)

            # Resize and pad the cropped region to 256x256
            resized_roi = cv2.resize(roi, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            top_padding = (max_size - new_height) // 2
            bottom_padding = max_size - new_height - top_padding
            left_padding = (max_size - new_width) // 2
            right_padding = max_size - new_width - left_padding
            padded_roi = cv2.copyMakeBorder(resized_roi, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=(255, 255, 255))

            # Convert to RGB for OCR processing
            roi_rgb = cv2.cvtColor(padded_roi, cv2.COLOR_GRAY2RGB)
            self.bounding_box_images.append(roi_rgb)

    def ocr_with_tesseract(self):
        for idx, roi_rgb in enumerate(self.bounding_box_images):
            pil_image = Image.fromarray(roi_rgb)
            extracted_text = pytesseract.image_to_string(pil_image, config='--psm 6')  # Adjust --psm mode if needed
            self.extracted_text_data[f"Bounding Box {idx + 1}"] = extracted_text.strip()

        # Save the extracted text to a JSON file
        with open("extracted_text.json", "w") as json_file:
            json.dump(self.extracted_text_data, json_file, indent=4)

    def clean_and_format_text(self):
        formatted_text = "\n".join([f"{key.replace('Bounding Box', '').strip()}: {value}"
                                    for key, value in self.extracted_text_data.items()])
        lines = formatted_text.split("\n")

        def clean_line(line):
            line = re.sub(r"[^\w\s.,:%()\-]", "", line)  # Clean unwanted chars
            line = re.sub(r"\s+", " ", line)  # Normalize spaces
            return line.strip()

        cleaned_lines = [clean_line(line) for line in lines if clean_line(line)]
        filtered_lines = [line for line in cleaned_lines if len(line.split()) > 1]

        return "\n".join(filtered_lines)



    def process_prescription(self):
        self.load_and_preprocess_image()
        self.apply_dilation()
        self.detect_contours()
        self.generate_bounding_boxes()
        self.merge_bounding_boxes()
        self.remove_overlapping_boxes()
        self.crop_and_resize_boxes()
        self.ocr_with_tesseract()

    def analyze_with_gemini(self, cleaned_text):
        if not self.gemini_key:
            raise ValueError("GEMINI_API_KEY not found. Please add it to the .env file.")

        genai.configure(api_key=self.gemini_key)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')

        prompt = f"""
        You are an AI assistant designed to analyze and structure medical prescription text. 
        Your task is to extract and correct relevant data from the input below.

        Instructions:
        1. Identify relevant information such as clinic name, address, doctor’s name, etc.
        2. Extract all prescribed medications, including their names, dosages, and frequencies, 
           ensuring the information is accurate and consistent with common medical and pharmaceutical standards.
           - Correct any abbreviations or errors in medication names.
           - Standardize units (e.g., "mg", "tabs", "mL").
           - Expand frequency abbreviations (e.g., "QD" → "once daily", "TID" → "three times daily").
           - Ensure medication dosages are correctly formatted (e.g., "100mg" → "100 mg").

        3. Ignore irrelevant or nonsensical text.
        4. Provide the result in a clean and structured format suitable for further processing. 
           Ensure clarity and logical organization.
        5. Write this at the begining "A human review by a qualified medical professional is
           necessary before dispensing any medication."  
        Input Text:
        {cleaned_text}
        """

        response = gemini_model.generate_content(prompt)
        self.gemini_response = response.text.strip()


        print("Formatted Response:")
        print(response.text)    

    def save_gemini_output_to_pdf(self, output_path="gemini_output.pdf"):
        if not self.gemini_response:
            raise ValueError("No Gemini response available. Run analyze_with_gemini() first.")
        
        cleaned_response = re.sub(r'\*+', '', self.gemini_response).strip()


        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Add Gemini response to PDF
        pdf.multi_cell(0, 10, cleaned_response)
        pdf.output(output_path)
        print(f"Gemini output saved to {output_path}")

    def process_prescription_with_analysis_and_pdf(self):
        self.process_prescription()  # Runs the OCR pipeline
        cleaned_text = self.clean_and_format_text()
        print("Cleaned Text:", cleaned_text)

        # Analyze the cleaned text with Gemini
        self.analyze_with_gemini(cleaned_text)

        # Save Gemini response to PDF
        self.save_gemini_output_to_pdf()  


# Usage
image_url = 'https://res.cloudinary.com/dvnf8yvsg/image/upload/v1733786371/otpdit20ub3afrjcrqw8.png'
processor = PrescriptionProcessor(image_url)
processor.process_prescription_with_analysis_and_pdf()
