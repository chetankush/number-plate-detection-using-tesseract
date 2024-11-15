# import cv2
# import numpy as np
# from pathlib import Path

# def initialize_cascade():
#     # Load the cascade classifier for license plate detection
#     cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
#     if not Path(cascade_path).exists():
#         raise FileNotFoundError("Cascade classifier file not found. Please download it from OpenCV repository.")
#     return cv2.CascadeClassifier(cascade_path)

# def detect_license_plates(frame, plate_cascade):
#     # Convert frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # Detect license plates
#     plates = plate_cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(25, 25)
#     )
    
#     return plates

# def draw_plates(frame, plates):
#     # Draw rectangles around detected license plates
#     for (x, y, w, h) in plates:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green rectangle
    
#     return frame

# def main():
#     try:
#         # Initialize the cascade classifier
#         plate_cascade = initialize_cascade()
        
#         # Initialize webcam
#         cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
        
#         if not cap.isOpened():
#             raise Exception("Could not open webcam")
        
#         print("License plate detection started. Press 'q' to quit.")
        
#         while True:
#             # Read frame from webcam
#             ret, frame = cap.read()
#             if not ret:
#                 print("Failed to grab frame")
#                 break
            
#             # Detect license plates
#             plates = detect_license_plates(frame, plate_cascade)
            
#             # Draw rectangles around detected plates
#             frame = draw_plates(frame, plates)
            
#             # Display the number of plates detected
#             cv2.putText(frame, f'Plates detected: {len(plates)}', 
#                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
#                        1, (0, 255, 0), 2)
            
#             # Show the frame
#             cv2.imshow('License Plate Detection', frame)
            
#             # Break the loop if 'q' is pressed
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
        
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
    
#     finally:
#         # Release resources
#         if 'cap' in locals():
#             cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()


import cv2
import numpy as np
from pathlib import Path
import pytesseract
from datetime import datetime

# Set the tesseract path - MODIFY THIS PATH to match your installation directory
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def initialize_cascade():
    cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
    if not Path(cascade_path).exists():
        raise FileNotFoundError("Cascade classifier file not found. Please download it from OpenCV repository.")
    return cv2.CascadeClassifier(cascade_path)

def detect_license_plates(frame, plate_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(25, 25)
    )
    return plates, gray

def process_plate_region(gray, x, y, w, h):
    # Extract the region of interest (license plate)
    plate_region = gray[y:y+h, x:x+w]
    
    # Preprocessing for better OCR
    plate_region = cv2.resize(plate_region, None, fx=2, fy=2)
    plate_region = cv2.bilateralFilter(plate_region, 11, 17, 17)
    plate_region = cv2.threshold(plate_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    try:
        # Perform OCR with additional error handling
        text = pytesseract.image_to_string(
            plate_region, 
            config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )
        return text.strip()
    except Exception as e:
        print(f"OCR Error: {str(e)}")
        return ""

def save_plate_text(text, confidence_threshold=0.6):
    if len(text) >= 4 and any(c.isalnum() for c in text):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open("detected_plates.txt", "a") as file:
                file.write(f"{timestamp} - Plate: {text}\n")
            return True
        except Exception as e:
            print(f"Error saving to file: {str(e)}")
    return False

def draw_plates(frame, plates, texts):
    for (x, y, w, h), text in zip(plates, texts):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if text:
            cv2.putText(frame, text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def main():
    try:
        # Verify Tesseract installation
        try:
            pytesseract.get_tesseract_version()
            print("Tesseract is properly installed.")
        except Exception as e:
            print(f"Tesseract Error: {str(e)}")
            print("Please verify Tesseract installation and path.")
            return

        plate_cascade = initialize_cascade()
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise Exception("Could not open webcam")
        
        print("License plate detection started. Press 'q' to quit.")
        
        with open("detected_plates.txt", "w") as file:
            file.write("=== License Plate Detection Log ===\n")
        
        processed_plates = set()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            plates, gray = detect_license_plates(frame, plate_cascade)
            texts = []
            
            for (x, y, w, h) in plates:
                text = process_plate_region(gray, x, y, w, h)
                texts.append(text)
                
                if text and text not in processed_plates:
                    if save_plate_text(text):
                        processed_plates.add(text)
            
            frame = draw_plates(frame, plates, texts)
            
            cv2.putText(frame, f'Plates detected: {len(plates)}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            
            cv2.imshow('License Plate Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()