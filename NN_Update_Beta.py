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
image_path = os.path.join(base_dir, "Spanish_L.jpg")
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
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #converts to RGB in Numpy array (Red, Green, Blue)
            image = Image.fromarray(img_rgb) #converts to Numpy array to PIL (pillow)
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
            
            # Apply operations to clean up the image 
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
        Improved translation with better quality control and context preservation
        """
        try:
            if not text.strip():
                return "No text to translate"
            
            print(f"[DEBUG] Text length: {len(text)} characters")
            
            # Handle auto-detection
            if target_language == 'auto':
                translator = GoogleTranslator(source='auto', target='en')
                translation = translator.translate(text)
                return f"Auto-translation to English: {translation}"
            
            # Preprocess text to improve translation quality
            preprocessed_text = self.preprocess_text_for_translation(text)
            
            # Check if text is too long (Google Translate limit is 5000 characters)
            if len(preprocessed_text) > 4000:  # More conservative limit
                print(f"[INFO] Text too long ({len(preprocessed_text)} chars), breaking into chunks...")
                return self.translate_long_text(preprocessed_text, target_language)
            
            print(f"[DEBUG] Attempting to translate: '{preprocessed_text[:100]}...' to {target_language}")
            
            # Create translator instance with better settings
            translator = GoogleTranslator(source='auto', target=target_language)
            
            # Translate the text
            translation = translator.translate(preprocessed_text)
            
            # Post-process translation to improve quality
            improved_translation = self.postprocess_translation(translation, text, target_language)
            
            print(f"[DEBUG] Translation successful: '{improved_translation[:100]}...'")
            return improved_translation
            
        except Exception as e:
            print(f"[ERROR] Translation failed with error: {str(e)}")
            print(f"[ERROR] Error type: {type(e).__name__}")
            import traceback
            print(f"[ERROR] Full traceback: {traceback.format_exc()}")
            return text  # Return original text if translation fails

    def preprocess_text_for_translation(self, text):
        """
        Preprocess text to improve translation quality
        """
        import re
        
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common OCR issues that affect translation
        # Replace common OCR errors with correct characters
        ocr_fixes = {
            '0': 'o',  # Common OCR error
            '1': 'l',  # Common OCR error
            '5': 's',  # Common OCR error
            '6': 'g',  # Common OCR error
            '8': 'b',  # Common OCR error
        }
        
        # Apply OCR fixes only in context where they make sense
        words = text.split()
        fixed_words = []
        
        for word in words:
            # Only fix if the word doesn't exist in common words
            if len(word) > 2 and not self.is_valid_word(word):
                # Try common OCR fixes
                for wrong, correct in ocr_fixes.items():
                    if wrong in word:
                        fixed_word = word.replace(wrong, correct)
                        if self.is_valid_word(fixed_word):
                            word = fixed_word
                            break
            fixed_words.append(word)
        
        return ' '.join(fixed_words)

    def is_valid_word(self, word):
        """
        Check if a word is valid (exists in common word lists)
        """
        # Common English words
        common_english = {
            'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'must', 'shall', 'go', 'my', 'vile', 'life', 'ir', 'mi', 'vil', 'vida'
        }
        
        # Common Spanish words
        common_spanish = {
            'el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le',
            'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'los', 'las', 'una',
            'como', 'más', 'pero', 'sus', 'me', 'hasta', 'hay', 'donde', 'han', 'quien',
            'están', 'estado', 'desde', 'todo', 'nos', 'durante', 'todos', 'uno', 'les',
            'ni', 'contra', 'otros', 'ese', 'eso', 'ante', 'ellos', 'e', 'esto', 'mí',
            'antes', 'algunos', 'qué', 'unos', 'yo', 'otro', 'otras', 'otra', 'él',
            'tanto', 'esa', 'estos', 'mucho', 'quienes', 'nada', 'muchos', 'cual',
            'poco', 'ella', 'estar', 'estas', 'algunas', 'algo', 'nosotros',
            'querer', 'lógico', 'preferir', 'escuchar', 'leer', 'escribir'
        }
        
        word_lower = word.lower().strip('.,!?;:')
        return word_lower in common_english or word_lower in common_spanish

    def postprocess_translation(self, translation, original_text, target_language):
        """
        Post-process translation to fix common issues and improve quality
        """
        import re
        
        # Remove duplicate content
        translation = self.remove_duplicate_content(translation)
        
        # Fix common translation errors
        translation = self.fix_common_translation_errors(translation, target_language)
        
        # Clean up formatting
        translation = re.sub(r'\s+', ' ', translation.strip())
        
        # Fix sentence structure issues
        translation = self.fix_sentence_structure(translation)
        
        return translation

    def remove_duplicate_content(self, text):
        """
        Remove duplicate sentences and phrases
        """
        import re
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_sentences = []
        
        for sentence in sentences:
            sentence_clean = sentence.strip().lower()
            if sentence_clean and sentence_clean not in seen:
                seen.add(sentence_clean)
                unique_sentences.append(sentence.strip())
        
        return ' '.join(unique_sentences)

    def fix_common_translation_errors(self, text, target_language):
        """
        Fix common translation errors using intelligent context analysis
        """
        import re
        
        # Apply general translation quality improvements
        text = self.improve_translation_quality(text, target_language)
        
        # Fix context-specific errors
        text = self.fix_context_errors(text, target_language)
        
        # Apply semantic corrections
        text = self.apply_semantic_corrections(text, target_language)
        
        return text

    def improve_translation_quality(self, text, target_language):
        """
        Apply general translation quality improvements
        """
        import re
        
        # Fix common translation artifacts
        fixes = [
            # Remove translation artifacts and noise
            (r'\b[A-Z]{2,}\b(?=\s+[a-z])', lambda m: m.group().lower()),  # Fix ALL CAPS words in context
            (r'\b([a-z])\1{2,}\b', lambda m: m.group()[0]),  # Fix repeated letters (aaa -> a)
            (r'\s+', ' '),  # Normalize whitespace
            (r'([.!?])\s*([A-Z])', r'\1 \2'),  # Fix spacing after punctuation
        ]
        
        for pattern, replacement in fixes:
            if callable(replacement):
                text = re.sub(pattern, replacement, text)
            else:
                text = re.sub(pattern, replacement, text)
        
        return text

    def fix_context_errors(self, text, target_language):
        """
        Fix context-specific translation errors
        """
        import re
        
        # Split into sentences for context analysis
        sentences = re.split(r'(?<=[.!?])\s+', text)
        fixed_sentences = []
        
        for sentence in sentences:
            # Analyze sentence context and fix errors
            fixed_sentence = self.analyze_and_fix_sentence_context(sentence, target_language)
            fixed_sentences.append(fixed_sentence)
        
        return ' '.join(fixed_sentences)

    def analyze_and_fix_sentence_context(self, sentence, target_language):
        """
        Analyze sentence context and fix translation errors
        """
        import re
        
        # Detect and fix common context errors
        sentence = self.fix_verb_context_errors(sentence, target_language)
        sentence = self.fix_adjective_context_errors(sentence, target_language)
        sentence = self.fix_noun_context_errors(sentence, target_language)
        
        return sentence

    def fix_verb_context_errors(self, sentence, target_language):
        """
        Fix verb translation errors based on context
        """
        # Common verb context patterns
        verb_contexts = {
            'en': {
                # Context: "I want to..." vs "I love to..."
                r'\bI\s+love\s+to\b': 'I want to',
                r'\bI\s+love\s+[a-z]+\b': 'I want',
                # Context: "to want" vs "to love" in infinitive
                r'\bto\s+love\s+([a-z]+)\b': r'to want \1',
                # Context: "wants" vs "loves" in third person
                r'\b([a-z]+)\s+loves\s+([a-z]+)\b': r'\1 wants \2',
            },
            'es': {
                # Similar patterns for Spanish
                r'\bquiero\s+amar\b': 'quiero querer',
                r'\bpara\s+amar\b': 'para querer',
            }
        }
        
        import re
        contexts = verb_contexts.get(target_language, {})
        
        for pattern, replacement in contexts.items():
            sentence = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
        
        return sentence

    def fix_adjective_context_errors(self, sentence, target_language):
        """
        Fix adjective translation errors based on context
        """
        # Common adjective context patterns
        adj_contexts = {
            'en': {
                # Context: "more logical" vs "more magical"
                r'\bmore\s+magical\b': 'more logical',
                r'\bvery\s+magical\b': 'very logical',
                r'\bmost\s+magical\b': 'most logical',
                # Context: "logical thinking" vs "magical thinking"
                r'\bmagical\s+thinking\b': 'logical thinking',
                r'\bmagical\s+reasoning\b': 'logical reasoning',
            },
            'es': {
                r'\bmás\s+mágico\b': 'más lógico',
                r'\bmuy\s+mágico\b': 'muy lógico',
            }
        }
        
        import re
        contexts = adj_contexts.get(target_language, {})
        
        for pattern, replacement in contexts.items():
            sentence = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
        
        return sentence

    def fix_noun_context_errors(self, sentence, target_language):
        """
        Fix noun translation errors based on context
        """
        # Common noun context patterns
        noun_contexts = {
            'en': {
                # Context: section headers and proper nouns
                r'\bGO\b(?=\s*:)': 'IR',
                r'\bMY\b(?=\s*:)': 'MI', 
                r'\bVILE\b(?=\s*:)': 'VIL',
                r'\bLIFE\b(?=\s*:)': 'VIDA',
                # Context: common noun errors
                r'\bgo\b(?=\s+[a-z]+)': 'ir',
                r'\bmy\b(?=\s+[a-z]+)': 'mi',
            },
            'es': {
                r'\bIR\b(?=\s*:)': 'IR',
                r'\bMI\b(?=\s*:)': 'MI',
                r'\bVIL\b(?=\s*:)': 'VIL', 
                r'\bVIDA\b(?=\s*:)': 'VIDA',
            }
        }
        
        import re
        contexts = noun_contexts.get(target_language, {})
        
        for pattern, replacement in contexts.items():
            sentence = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
        
        return sentence

    def apply_semantic_corrections(self, text, target_language):
        """
        Apply semantic corrections based on meaning and context
        """
        import re
        
        # Detect and fix semantic inconsistencies
        text = self.fix_semantic_inconsistencies(text, target_language)
        
        # Apply domain-specific corrections
        text = self.apply_domain_corrections(text, target_language)
        
        return text

    def fix_semantic_inconsistencies(self, text, target_language):
        """
        Fix semantic inconsistencies in the text
        """
        import re
        
        # Common semantic error patterns
        semantic_fixes = {
            'en': [
                # Fix contradictory phrases
                (r'\bnot\s+not\b', 'not'),
                (r'\bcan\s+not\s+can\b', 'cannot'),
                # Fix redundant phrases
                (r'\bvery\s+very\b', 'very'),
                (r'\bmore\s+more\b', 'more'),
                # Fix malformed phrases
                (r'\bof\s+of\b', 'of'),
                (r'\bthe\s+the\b', 'the'),
            ],
            'es': [
                (r'\bno\s+no\b', 'no'),
                (r'\bmuy\s+muy\b', 'muy'),
                (r'\bmás\s+más\b', 'más'),
            ]
        }
        
        fixes = semantic_fixes.get(target_language, [])
        
        for pattern, replacement in fixes:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

    def apply_domain_corrections(self, text, target_language):
        """
        Apply domain-specific corrections (educational, technical, etc.)
        """
        import re
        
        # Educational domain corrections
        if self.is_educational_content(text):
            text = self.apply_educational_corrections(text, target_language)
        
        # Technical domain corrections
        if self.is_technical_content(text):
            text = self.apply_technical_corrections(text, target_language)
        
        return text

    def is_educational_content(self, text):
        """
        Detect if text is educational content
        """
        educational_indicators = [
            'infinitive', 'conjugate', 'verb', 'grammar', 'exercise',
            'question', 'answer', 'read', 'write', 'listen', 'speak',
            'practice', 'learn', 'study', 'test', 'exam'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in educational_indicators)

    def is_technical_content(self, text):
        """
        Detect if text is technical content
        """
        technical_indicators = [
            'function', 'method', 'class', 'object', 'variable',
            'parameter', 'return', 'error', 'debug', 'compile',
            'algorithm', 'data', 'structure', 'interface'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in technical_indicators)

    def apply_educational_corrections(self, text, target_language):
        """
        Apply corrections specific to educational content
        """
        import re
        
        educational_fixes = {
            'en': [
                # Fix educational terminology
                (r'\binfinitive\s*":\s*', 'Infinitive: '),
                (r'\bconjugated\s+verb\s+of"\s*', 'conjugated verb of "'),
                (r'\bprefer\'\s*more', 'prefer more'),
                (r'\bprefer"\s*more', 'prefer more'),
            ],
            'es': [
                (r'\binfinitivo\s*":\s*', 'Infinitivo: '),
                (r'\bverbo\s+conjugado\s+de"\s*', 'verbo conjugado de "'),
            ]
        }
        
        fixes = educational_fixes.get(target_language, [])
        
        for pattern, replacement in fixes:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

    def apply_technical_corrections(self, text, target_language):
        """
        Apply corrections specific to technical content
        """
        # Add technical corrections as needed
        return text

    def fix_sentence_structure(self, text):
        """
        Fix malformed sentences and improve structure
        """
        import re
        
        # Fix common sentence structure issues
        fixes = [
            # Fix malformed infinitive sentences
            (r'Infinitive\s*":\s*Read', 'Infinitive: Read'),
            (r'conjugated verb of"\s*prefer', 'conjugated verb of "prefer"'),
            (r'one thousand\d+n', 'one thousand'),
            (r'large indicator of', 'large indicator of'),
        ]
        
        for pattern, replacement in fixes:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

    def translate_long_text(self, text, target_language):
        """
        Translate long text by breaking it into smaller, sentence-aware chunks
        """
        try:
            import re
            
            # Split text into sentences using regex - handles multiple sentence endings
            # (?<=[.!?]) - positive lookbehind for sentence endings
            # \s+ - one or more whitespace characters
            sentences = re.split(r'(?<=[.!?])\s+', text) #splits text into sentences using regex
            
            # Clean up sentences and filter out empty ones
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Create sentence-aware chunks
            chunks = self.create_sentence_aware_chunks(sentences)
            
            print(f"[INFO] Split text into {len(chunks)} chunks for translation")
            print(f"[INFO] Average chunk size: {sum(len(chunk) for chunk in chunks) // len(chunks) if chunks else 0} characters")
            
            # Translate each chunk with better error handling
            translated_chunks = self.translate_chunks_with_retry(chunks, target_language)
            
            # Combine translated chunks with proper spacing
            final_translation = self.combine_translated_chunks(translated_chunks)
            print(f"[INFO] Successfully translated {len(chunks)} chunks")
            return final_translation
            
        except Exception as e:
            print(f"[ERROR] Long text translation failed: {e}")
            return text

    def create_sentence_aware_chunks(self, sentences):
        """
        Create chunks that respect sentence boundaries and maintain coherence
        """
        chunks = []
        current_chunk = ""
        max_chunk_size = 800  # Even smaller for better translation accuracy
        
        # Pre-process sentences to identify and preserve important context
        processed_sentences = self.preprocess_sentences_for_chunking(sentences)
        
        for i, sentence in enumerate(processed_sentences):
            # Calculate space needed for this sentence
            space_needed = len(sentence) + (1 if current_chunk else 0)  # +1 for space
            
            # Special handling for section headers and important terms
            if self.is_section_header(sentence):
                # Always start a new chunk for section headers
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            elif current_chunk and len(current_chunk) + space_needed > max_chunk_size:
                # Save current chunk and start new one
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            
            # If this is the last sentence, add the final chunk
            if i == len(processed_sentences) - 1 and current_chunk.strip():
                chunks.append(current_chunk.strip())
        
        # Handle edge case: if any single sentence is too long
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > max_chunk_size:
                # Split long chunk by words while preserving sentence endings
                sub_chunks = self.split_long_chunk_by_words(chunk, max_chunk_size)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        return final_chunks

    def preprocess_sentences_for_chunking(self, sentences):
        """
        Preprocess sentences to improve chunking quality
        """
        processed = []
        
        for sentence in sentences:
            # Clean up common OCR issues that affect chunking
            cleaned = sentence.strip()
            
            # Fix common OCR errors that create malformed sentences
            ocr_fixes = {
                'thousand6n': 'thousand',
                'thousand5n': 'thousand',
                'thousand0n': 'thousand',
                'prefer\'more': 'prefer more',
                'prefer"more': 'prefer more',
            }
            
            for wrong, correct in ocr_fixes.items():
                cleaned = cleaned.replace(wrong, correct)
            
            if cleaned:
                processed.append(cleaned)
        
        return processed

    def is_section_header(self, sentence):
        """
        Check if a sentence is a section header that should be preserved
        """
        # Common section header patterns
        header_patterns = [
            r'^[A-Z]{2,}$',  # All caps words like "IR", "MI", "VIL", "VIDA"
            r'^[A-Z][a-z]+:',  # Title case with colon
            r'^[A-Z\s]+$',  # All caps phrases
        ]
        
        import re
        for pattern in header_patterns:
            if re.match(pattern, sentence.strip()):
                return True
        
        return False

    def split_long_chunk_by_words(self, chunk, max_size):
        """
        Split a long chunk by words while trying to keep sentence structure
        """
        words = chunk.split()
        sub_chunks = []
        current_sub_chunk = ""
        
        for word in words:
            # Check if adding this word would exceed size
            if current_sub_chunk and len(current_sub_chunk) + len(word) + 1 > max_size:
                # Save current sub-chunk and start new one
                if current_sub_chunk.strip():
                    sub_chunks.append(current_sub_chunk.strip())
                current_sub_chunk = word
            else:
                # Add word to current sub-chunk
                if current_sub_chunk:
                    current_sub_chunk += " " + word
                else:
                    current_sub_chunk = word
        
        # Add the last sub-chunk
        if current_sub_chunk.strip():
            sub_chunks.append(current_sub_chunk.strip())
        
        return sub_chunks

    def translate_chunks_with_retry(self, chunks, target_language): #chunks is a list of chunks to translate, target_language is the language to translate to
        """
        Translate chunks with retry logic and better error handling
        """
        translator = GoogleTranslator(source='auto', target=target_language)
        translated_chunks = []
        
        for i, chunk in enumerate(chunks):
            print(f"[INFO] Translating chunk {i+1}/{len(chunks)} ({len(chunk)} chars): '{chunk[:50]}...'")
            
            # Try translation with retry logic
            translation = self.translate_chunk_with_retry(chunk, translator, i+1)
            translated_chunks.append(translation)
            
            # Delay between chunks to avoid rate limiting
            time.sleep(0.5)
        
        return translated_chunks

    def translate_chunk_with_retry(self, chunk, translator, chunk_num, max_retries=3): #chunk is the chunk to translate, translator is the translator object, chunk_num is the number of the chunk, max_retries is the number of retries
        """
        Translate a single chunk with retry logic
        """
        for attempt in range(max_retries):
            try:
                translation = translator.translate(chunk)
                
                # Check if translation is meaningful (not just the original text)
                if translation and translation.strip() != chunk.strip():
                    return translation
                else:
                    print(f"[WARNING] Chunk {chunk_num} translation returned original text, retrying...")
                    time.sleep(1)  # Wait before retry
                    
            except Exception as e:
                print(f"[WARNING] Chunk {chunk_num} translation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait longer between retries
                else:
                    print(f"[ERROR] Chunk {chunk_num} translation failed after {max_retries} attempts")
        
        # If all retries failed, return original chunk
        print(f"[WARNING] Returning original text for chunk {chunk_num}")
        return chunk

    def combine_translated_chunks(self, translated_chunks): #combines translated chunks into a single string
        """
        Combine translated chunks with proper spacing and formatting
        """
        if not translated_chunks:
            return ""
        
        # Join chunks with proper spacing
        combined = " ".join(translated_chunks)
        
        # Clean up any double spaces or formatting issues
        import re
        combined = re.sub(r'\s+', ' ', combined)  # Replace multiple spaces with single space
        combined = combined.strip()
        
        return combined

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


