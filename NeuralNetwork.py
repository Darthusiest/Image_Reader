import pytesseract
from PIL import Image
import nltk
nltk.download('words')
from nltk.corpus import words
import difflib
import os

image_path = r"C:\Users\jakep\OneDrive\Documents\Python\Neural Networks\stop_sign.jpg.jpg"
print("Exists:", os.path.exists(image_path))
print("Absolute path used:", image_path)

if not os.path.exists(image_path):
    print("Image file does not exist.")
    exit()
else:
    image = Image.open(image_path)
    print("Image opened successfully.")

class NeuralNetwork():
    def __init__(self):
        pass
    def extract_image(self, image_path):
        image = Image.open(image_path) #still working on image path
        text = pytesseract.image_to_string(image) #grab the text from the image
        print(f"Raw text extracted from image: '{text.strip()}'") #raw text extracted from image
        words = text.split() #split the text into words
        
        #validate word
        valid_words = []
        for word in words:
            clean_word = ''.join(char for char in word if char.isalpha())
            if clean_word and self.validate_word(clean_word): #if clean & valid add to list
                    valid_words.append(clean_word) #add to list
            else:
                corrected = self.correct_word(clean_word) #correct the word
                valid_words.append(corrected) #add to list
        return valid_words
    
    def validate_word(self, word):
        return word.lower() in words.words() #check if word is in word list

    def correct_word(self, word): #correct the word if it's not in the word list
        word_list = words.words() #get word list
        matches = difflib.get_close_matches(word.lower(), word_list, n = 1, cutoff = 0.8) #get close matches, n = 1 for first match, cutoff = 80% similarity
        return matches[0] if matches else word #return closest match or original word


nn = NeuralNetwork()
result = nn.extract_image(image_path)
print("[RESULT] Final output words:", result)