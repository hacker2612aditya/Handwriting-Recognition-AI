import cv2
import pytesseract
import numpy as np
import time

# Set path to tesseract.exe on Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to improve image for better OCR
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    return dilated

cap = cv2.VideoCapture(0)

# Variables for timed display
last_text = ""
display_text = ""
last_update_time = 0
DISPLAY_DURATION = 3  # seconds

print("📷 Live OCR started. Hold handwriting to camera. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    processed = preprocess_image(frame)

    # Use Tesseract with better config
    custom_config = r'--oem 1 --psm 6'
    text = pytesseract.image_to_string(processed, lang='eng', config=custom_config)
    text = text.strip().replace('\n', ' ')

    current_time = time.time()

    # Update display text only if it's new and not empty
    if text and text != last_text:
        last_text = text
        display_text = text
        last_update_time = current_time

    # Show text if within display duration
    if current_time - last_update_time <= DISPLAY_DURATION:
        cv2.putText(frame, display_text[:50], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display video
    cv2.imshow("✍️ Handwriting Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
