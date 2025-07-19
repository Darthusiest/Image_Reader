from tkinter import Y
import pytesseract #pull text from a image
import numpy as np #image processing
from PIL import Image # image processing
import nltk #library of english words
import difflib #compare words
import os #file path
import cv2 #image processing
from deep_translator import GoogleTranslator #translate text
import time #delay
from spellchecker import SpellChecker #multi-language spell checker
import pdf2image #convert PDF to images

#png, jpg conversion
#special char handling  john@something.com == john@something.com
#context of the image (the words)


# Download word list if not already downloaded
nltk.download('words')
from nltk.corpus import words as nltk_words

# Set up path to Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Tesseract-OCR\tesseract.exe"

# Path to your image
base_dir = os.path.dirname(__file__) # allows for img in same dir pathway (EX: Folder/NN_Update.py, stop.jpg)
image_path = os.path.join(base_dir, "Git_list.pdf")
print("Exists:", os.path.exists(image_path))
print("Absolute path used:", image_path) 

if not os.path.exists(image_path):
    print("[ERROR] Image file does not exist.")
    exit()

class NeuralNetwork:
    def __init__(self):
        pass

    def extract_image(self, file_path):
        try:
            # Check if file is PDF or image
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.pdf':
                print("[INFO] Processing PDF file...")
                return self.extract_from_pdf(file_path)
            else:
                print("[INFO] Processing image file...")
                return self.extract_from_image(file_path)
                
        except Exception as e:
            print(f"[ERROR] Failed to process file: {e}")
            return []

    def extract_from_pdf(self, pdf_path):
        """
        Extract text from PDF by converting pages to images first
        """
        try:
            # Convert PDF pages to images
            print("[INFO] Converting PDF pages to images...")
            
            # Try to find Poppler in common installation paths
            poppler_path = None
            possible_paths = [ #poppler is a library that converts PDF to images
                r"C:\Program Files\poppler\Library\bin",
                r"C:\Program Files (x86)\poppler\Library\bin", 
                r"C:\poppler\bin",
                os.path.expanduser(r"~\AppData\Local\Programs\poppler\bin") #this is a path to the poppler library
            ]
            
            for path in possible_paths: #check if the path exists
                if os.path.exists(path):
                    poppler_path = path
                    print(f"[INFO] Found Poppler at: {poppler_path}")
                    break
            
            if poppler_path:
                images = pdf2image.convert_from_path(pdf_path, poppler_path=poppler_path) #convert PDF to images
            else:
                print("[WARNING] Poppler path not found, trying default...")
                images = pdf2image.convert_from_path(pdf_path)
                
            print(f"[INFO] PDF has {len(images)} pages")
            
            all_text = []
            
            for i, image in enumerate(images):
                print(f"[INFO] Processing page {i+1}/{len(images)}...")
                
                # Convert PIL image to OpenCV format
                img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Preprocess the image
                img_cv = self.preprocess_image_for_ocr(img_cv)
                
                # Convert back to PIL for OCR
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(img_rgb)
                
                # Extract text from this page
                page_text = self.extract_text_with_improved_ocr(pil_image)
                
                if page_text.strip():
                    all_text.append(page_text.strip())
                    print(f"[INFO] Page {i+1} text: '{page_text.strip()[:100]}...'")
                else:
                    print(f"[WARN] No text found on page {i+1}")
            
            # Combine all page text
            combined_text = ' '.join(all_text)
            
            if not combined_text.strip():
                print("[WARN] No text could be extracted from PDF.")
                return []
            
            # Detect language and apply spelling correction
            corrected_text = self.correct_text_by_language(combined_text.strip())
            return corrected_text
            
        except Exception as e:
            print(f"[ERROR] PDF processing failed: {e}")
            return []

    def extract_from_image(self, image_path):
        """
        Extract text from image file (original functionality)
        """
        try:
            img = cv2.imread(image_path) #load img into Numpy array, EX: np.array([(255,255,255), (255,255,255), (255,255,255)])
            if img is None:
                print("[ERROR] Failed to load image from path:", image_path)
                return []

            # Preprocess image for better OCR
            img = self.preprocess_image_for_ocr(img)
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img_rgb)
            print("[OK] Image opened and preprocessed successfully.")
        except Exception as e:
            print("[ERROR] Failed to open image:", e)
            return []

        # Extract text from image with improved OCR settings
        # Try multiple OCR configurations for better results
        text = self.extract_text_with_improved_ocr(image)
        print("[INFO] Raw text extracted from image:")
        print("'" + text.strip() + "'")
        print()

        if not text.strip():
            #if no text, return empty list
            print("[WARN] No text could be extracted from the image.")
            return []

        # Detect language and apply appropriate spelling correction
        corrected_text = self.correct_text_by_language(text.strip())
        return corrected_text

    def extract_text_with_improved_ocr(self, image):
        """
        Extract text using multiple OCR configurations for better accuracy
        """
        # Try different OCR configurations
        configs = [
            '--oem 3 --psm 6',  # Default: Assume uniform block of text
            '--oem 3 --psm 3',  # Fully automatic page segmentation
            '--oem 3 --psm 4',  # Assume single column of text
            '--oem 3 --psm 8',  # Single word
            '--oem 1 --psm 6',  # Legacy engine with uniform block
        ]
        
        best_text = ""
        best_confidence = 0
        
        for config in configs:
            try:
                # Extract text with current configuration
                text = pytesseract.image_to_string(image, config=config)
                
                # Calculate a simple confidence score based on text quality
                confidence = self.calculate_text_confidence(text)
                
                print(f"[DEBUG] Config {config}: Confidence {confidence:.2f}")
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_text = text
                    
            except Exception as e:
                print(f"[WARNING] OCR config {config} failed: {e}")
                continue
        
        # If no good results, fall back to default
        if not best_text.strip():
            print("[WARNING] All OCR configs failed, using default")
            best_text = pytesseract.image_to_string(image)
        
        return best_text

    def calculate_text_confidence(self, text):
        """
        Calculate a simple confidence score for OCR text quality
        """
        if not text.strip():
            return 0
        
        score = 0
        words = text.split()
        
        # Higher score for more words
        score += len(words) * 0.1
        
        # Higher score for longer average word length (indicates real words)
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        score += avg_word_length * 0.2
        
        # Penalty for excessive special characters
        special_char_ratio = sum(1 for char in text if not char.isalnum() and not char.isspace()) / len(text) if text else 0
        score -= special_char_ratio * 10
        
        # some common words to fall back on
        common_words = ['the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'los', 'las', 'una', 'como', 'más', 'pero', 'sus', 'me', 'hasta', 'hay', 'donde', 'han', 'quien', 'están', 'estado', 'desde', 'todo', 'nos', 'durante', 'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'ese', 'eso', 'ante', 'ellos', 'e', 'esto', 'mí', 'antes', 'algunos', 'qué', 'unos', 'yo', 'otro', 'otras', 'otra', 'él', 'tanto', 'esa', 'estos', 'mucho', 'quienes', 'nada', 'muchos', 'cual', 'poco', 'ella', 'estar', 'estas', 'algunas', 'algo', 'nosotros']
        common_word_count = sum(1 for word in words if word.lower() in common_words)
        score += common_word_count * 0.5
        
        return max(0, score)

    def preprocess_image_for_ocr(self, img):
        """
        Preprocess image to improve OCR accuracy
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise on image
            blurred = cv2.GaussianBlur(gray, (1, 1), 0)
            
            # Apply thresholding to get binary image AKA (black and white)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #OTSU is a thresholding method that automatically finds the optimal threshold value
            
            # Apply morphological operations to clean up the image
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)) #turns into 1x1 matrix 

            #Dilation - this expands white areas
            #Erosion - this shrinks white areas
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel) #connects broken char's like 'i', fills gaps in letters like 'o'

            #After this closing text like H E L L O = Hello
            
            # Resize image if it's too small (improves OCR accuracy)
            height, width = cleaned.shape #tuple (height, width)
            if width < 800: #if width is less than 800 its to small, so we should scale(enlarge) the image
                scale_factor = 800 / width #Example img = 400px, 800/400 = 2, so we should scale the image by 2
                new_width = int(width * scale_factor) #400 * 2 = 800
                new_height = int(height * scale_factor) #200 * 2 = 400
                cleaned = cv2.resize(cleaned, (new_width, new_height), interpolation=cv2.INTER_CUBIC) #cv2.INTER_CUBIC gives better quality (let the function handle that)
            
            # Convert back to BGR for consistency
            result = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR) #converts to BGR (Blue, Green, Red)
            
            return result
            
        except Exception as e:
            print(f"[WARNING] Image preprocessing failed: {e}")
            return img  # Return original image if preprocessing fails

    def validate_word(self, word):
        return word.lower() in nltk_words.words()

    def correct_word(self, word):
        matches = difflib.get_close_matches(word.lower(), nltk_words.words(), n=1, cutoff=0.8)
        return matches[0] if matches else word

    def detect_language(self, text):
        """
        Detect the language of the given text using Google Translate
        """
        try:
            # First, check for Spanish indicators (more reliable for Spanish text)
            spanish_indicators = ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'los', 'las', 'una', 'como', 'más', 'pero', 'sus', 'me', 'hasta', 'hay', 'donde', 'han', 'quien', 'están', 'estado', 'desde', 'todo', 'nos', 'durante', 'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'ese', 'eso', 'ante', 'ellos', 'e', 'esto', 'mí', 'antes', 'algunos', 'qué', 'unos', 'yo', 'otro', 'otras', 'otra', 'él', 'tanto', 'esa', 'estos', 'mucho', 'quienes', 'nada', 'muchos', 'cual', 'poco', 'ella', 'estar', 'estas', 'algunas', 'algo', 'nosotros']
            
            text_lower = text.lower()
            spanish_count = sum(1 for word in spanish_indicators if word in text_lower)
            
            # If we find multiple Spanish words, it's likely Spanish
            if spanish_count >= 2:
                print(f"[DEBUG] Found {spanish_count} Spanish indicators, detecting as Spanish")
                return 'es'
            
            # Use Google Translator to detect language
            translator = GoogleTranslator(source='auto', target='en')
            translation = translator.translate(text)
            
            # If translation is very similar to original, it's likely English
            if text.lower().strip() == translation.lower().strip():
                print("[DEBUG] Translation matches original, detecting as English")
                return 'en'
            
            # If translation is significantly different, check if it's Spanish
            # by trying to translate back to Spanish
            try:
                translator_to_spanish = GoogleTranslator(source='en', target='es')
                back_translation = translator_to_spanish.translate(translation)
                
                # If back translation is similar to original, it was Spanish
                if len(text) > 10:  # Only for longer texts
                    similarity = self.calculate_similarity(text.lower(), back_translation.lower())
                    if similarity > 0.7:  # 70% similarity threshold
                        print(f"[DEBUG] Back translation similarity: {similarity:.2f}, detecting as Spanish")
                        return 'es'
            except:
                pass
            
            # Default to English for other languages
            print("[DEBUG] Defaulting to English")
            return 'en'
                
        except Exception as e:
            print(f"[WARNING] Language detection failed: {e}")
            return 'en'  # Default to English

    def calculate_similarity(self, text1, text2):
        """
        Calculate similarity between two texts using difflib
        """
        try:
            return difflib.SequenceMatcher(None, text1, text2).ratio()
        except:
            return 0.0

    def correct_text_by_language(self, text):
        """
        Use pyspellchecker for multi-language spell checking
        """
        if not text.strip():
            return text
            
        detected_lang = self.detect_language(text)
        print(f"[INFO] Detected language: {detected_lang}")
        
        return self.spell_check_text(text, detected_lang) #have a function for spell checking

    def spell_check_text(self, text, language):
        """
        Multi-language spell checking (removed manual corrections)
        """
        try:
            # Map language codes
            lang_mapping = {'en': 'en', 'es': 'es', 'fr': 'fr', 'de': 'de', 'pt': 'pt', 'ru': 'ru'}
            spell_lang = lang_mapping.get(language, 'en')
            
            # Use spell checker directly (no manual corrections first)
            spell = SpellChecker(language=spell_lang)
            
            # Process words with spell checker
            words = text.split()
            corrected_words = []
            
            for word in words:
                # Let spell checker handle everything
                clean_word = ''.join(char for char in word if char.isalpha())
                if not clean_word:
                    corrected_words.append(word)
                    continue
                    
                if clean_word.lower() in spell:
                    corrected_words.append(word)  # Word is correct
                else:
                    correction = spell.correction(clean_word)
                    if correction:
                        # Apply correction with case preservation
                        if clean_word.isupper():
                            corrected_word = correction.upper()
                        elif clean_word.istitle():
                            corrected_word = correction.title()
                        else:
                            corrected_word = correction
                        corrected_words.append(word.replace(clean_word, corrected_word))
                    else:
                        corrected_words.append(word)  # Keep original
            
            return ' '.join(corrected_words)
            
        except Exception as e:
            print(f"[WARNING] Spell check failed: {e}")
            return text

    def translate_text(self, text, target_language='en'):
        """
        Original Text
        ↓
        Split into chunks:
        ├── Chunk 1 (1800 chars) → Translate → English chunk 1
        ├── Chunk 2 (2100 chars) → Translate → English chunk 2
        └── Chunk 3 (1334 chars) → Translate → English chunk 3
        ↓
        Combine: "English chunk 1 + English chunk 2 + English chunk 3"
        ↓
        Final Result: Complete English translation
        Args:
            text (str): Text to translate
            target_language (str): Target language code (e.g., 'es' for Spanish, 'auto' for auto-detection)
        
        Returns:
            str: Translated text
        """
        try:
            if not text.strip():
                return "No text to translate"
            
            print(f"[DEBUG] Text length: {len(text)} characters")
            
            # Handle auto-detection
            if target_language == 'auto':
                # For auto-detection, translate to English and let the translator handle source detection
                translator = GoogleTranslator(source='auto', target='en')
                translation = translator.translate(text)
                return f"Auto-translation to English: {translation}"
            
            # Check if text is too long (Google Translate limit is 5000 characters)
            if len(text) > 4500:  # Leave some buffer
                print(f"[INFO] Text too long ({len(text)} chars), breaking into chunks...")
                return self.translate_long_text(text, target_language)
            
            print(f"[DEBUG] Attempting to translate: '{text[:100]}...' to {target_language}")
            
            # Create translator instance for specific target language
            # Use 'auto' as source to automatically detect the input language
            translator = GoogleTranslator(source='auto', target=target_language)
            
            # Translate the text
            translation = translator.translate(text)
            print(f"[DEBUG] Translation successful: '{translation[:100]}...'")
            return translation
            
        except Exception as e:
            print(f"[ERROR] Translation failed with error: {str(e)}")
            print(f"[ERROR] Error type: {type(e).__name__}")
            import traceback
            print(f"[ERROR] Full traceback: {traceback.format_exc()}")
            return text  # Return original text if translation fails

    def translate_long_text(self, text, target_language):
        """
        Translate long text by breaking it into chunks
        """
        try:
            # Split text into sentences or paragraphs
            sentences = text.split('. ')
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                # If adding this sentence would make chunk too long, start new chunk
                if len(current_chunk + sentence) > 4000:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        # Single sentence is too long, split it
                        words = sentence.split()
                        current_chunk = ""
                        for word in words:
                            if len(current_chunk + word) > 4000:
                                if current_chunk:
                                    chunks.append(current_chunk.strip())
                                    current_chunk = word
                                else:
                                    chunks.append(word[:4000])
                                    current_chunk = word[4000:]
                            else:
                                current_chunk += " " + word if current_chunk else word
                else:
                    current_chunk += ". " + sentence if current_chunk else sentence
            
            # Add the last chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            print(f"[INFO] Split text into {len(chunks)} chunks for translation")
            
            # Translate each chunk
            translator = GoogleTranslator(source='auto', target=target_language)
            translated_chunks = []
            
            for i, chunk in enumerate(chunks):
                print(f"[INFO] Translating chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
                try:
                    translation = translator.translate(chunk)
                    translated_chunks.append(translation)
                    time.sleep(0.5)  # Small delay between chunks
                except Exception as e:
                    print(f"[WARNING] Failed to translate chunk {i+1}: {e}")
                    translated_chunks.append(chunk)  # Keep original if translation fails
            
            # Combine translated chunks
            final_translation = ' '.join(translated_chunks)
            print(f"[INFO] Successfully translated {len(chunks)} chunks")
            return final_translation
            
        except Exception as e:
            print(f"[ERROR] Long text translation failed: {e}")
            return text

# Run the OCR and print result
nn = NeuralNetwork()
result = nn.extract_image(image_path)
print("[RESULT] Final output words:", result)

# User interaction for translation
if result:
    print("\n=== TRANSLATION OPTIONS ===")
    print("Would you like to translate this text? (y/n): ", end="")
    
    # Get user input
    user_choice = input().lower().strip()
    
    if user_choice in ['y', 'yes']:
        print("\n=== AVAILABLE LANGUAGES ===")
        print("Common language codes:")
        print("- 'en' for English")
        print("- 'es' for Spanish")
        print("- 'fr' for French") 
        print("- 'de' for German")
        print("- 'it' for Italian")
        print("- 'pt' for Portuguese")
        print("- 'ru' for Russian")
        print("- 'ja' for Japanese")
        print("- 'ko' for Korean")
        print("- 'zh' for Chinese")
        print("- 'ar' for Arabic")
        print("- 'hi' for Hindi")
        print("- 'auto' for auto-detection")
        print("\nYou can also try any other language code supported by Google Translate!")
        

        print("\nEnter the language code (e.g., 'es' for Spanish): ", end="")
        target_language = input().lower().strip()
        
        # Try to translate with the provided language code
        print(f"\n[TRANSLATING] Translating to {target_language}...")
        
        # Add a small delay to avoid rate limiting
        import time
        time.sleep(1)
        
        try:
            translated_text = nn.translate_text(result, target_language)
            if translated_text and translated_text != result:
                print(f"[FINAL OUTPUT] Translated text: {translated_text}")
            else:
                print(f"[WARNING] Translation returned original text or empty result")
                print(f"[FINAL OUTPUT] Original text: {result}")
        except Exception as e:
            print(f"[ERROR] Translation failed for language '{target_language}': {e}")
            print("Trying with default language (Spanish)...")
            time.sleep(1)  # Another delay before retry
            try:
                translated_text = nn.translate_text(result, 'es')
                if translated_text and translated_text != result:
                    print(f"[FINAL OUTPUT] Translated text: {translated_text}")
                else:
                    print(f"[WARNING] Spanish translation also returned original text")
                    print(f"[FINAL OUTPUT] Original text: {result}")
            except Exception as e2:
                print(f"[ERROR] Spanish translation also failed: {e2}")
                print(f"[FINAL OUTPUT] Original text: {result}")
    
    else:
        print("[INFO] No translation.")
        print(f"[FINAL OUTPUT] Original text: {result}")
else:
    print("[FINAL OUTPUT] No text extracted from image.")


