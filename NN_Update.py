import pytesseract #pull text from a image
from PIL import Image # image processing
import nltk #library of english words
import difflib
import os
import cv2

#png, jpg conversion
#translate to english using google translate api
#special char handling  john@something.com == john@something.com
#context of the image (the words)


# Download word list if not already downloaded
nltk.download('words')
from nltk.corpus import words as nltk_words

# Set up path to Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Tesseract-OCR\tesseract.exe"

# Path to your image
#
image_path = r"C:\Users\jakep\OneDrive\Documents\Python\Neural Networks\stop.jpg"
print("Exists:", os.path.exists(image_path))
print("Absolute path used:", image_path) 

if not os.path.exists(image_path):
    print("[ERROR] Image file does not exist.")
    exit()

class NeuralNetwork:
    def __init__(self):
        pass

    def extract_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #Otsu's thresholding

            temp_file = "temp_image.png"
            cv2.imwrite(temp_file, thresh)
            image = Image.open(temp_file)
            print("[OK] Image opened successfully.")
        except Exception as e:
            print("[ERROR] Failed to open image:", e)
            return []

        # Extract text from image
        #im strusting the text but should have safe gaurds for the text
        text = pytesseract.image_to_string(image)
        print("[INFO] Raw text extracted from image:")
        print("'" + text.strip() + "'")
        print()

        if not text.strip():
            print("[WARN] No text could be extracted from the image.")
            return []

        raw_words = text.split()
        final_words = []

        for word in raw_words:
            clean_word = ''.join(char for char in word if char.isalpha())
            if not clean_word:
                continue  # Skip empty or garbage words

            if self.validate_word(clean_word):
                final_words.append(clean_word)
            else:
                corrected = self.correct_word(clean_word)
                final_words.append(corrected)

        return final_words

    def validate_word(self, word):
        return word.lower() in nltk_words.words()

    def correct_word(self, word):
        matches = difflib.get_close_matches(word.lower(), nltk_words.words(), n=1, cutoff=0.8)
        return matches[0] if matches else word

# Run the OCR and print result
nn = NeuralNetwork()
result = nn.extract_image(image_path)
print("[RESULT] Final output words:", result)
