import numpy as np #image processing
from PIL import Image # image processing
import os #file path
import cv2 #image processing
from deep_translator import GoogleTranslator #translate text
import time #delay
import base64
import requests
import json
import tempfile

# For Vercel deployment, we'll use cloud-based OCR services
# You can use services like:
# - Google Cloud Vision API
# - Azure Computer Vision
# - AWS Textract
# - OCR.space API (free tier available)

class NeuralNetwork:
    def __init__(self):
        # For Vercel, we'll use OCR.space API as it has a free tier
        # You can get a free API key from https://ocr.space/ocrapi
        self.ocr_api_key = os.getenv('OCR_API_KEY', '')  # Set this in Vercel environment variables
        self.ocr_api_url = 'https://api.ocr.space/parse/image'
        pass

    def extract_image(self, file_path):
        """
        Extract text from image file using cloud OCR service
        """
        try:
            # Check if we have API key for cloud OCR
            if self.ocr_api_key:
                return self.extract_with_cloud_ocr(file_path)
            else:
                # Fallback to local OCR if available
                return self.extract_with_local_ocr(file_path)
                
        except Exception as e:
            print(f"[ERROR] Failed to process file: {e}")
            return []

    def extract_with_cloud_ocr(self, file_path):
        """
        Extract text using OCR.space API
        """
        try:
            # Read file as base64
            with open(file_path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
            
            # Prepare API request
            payload = {
                'apikey': self.ocr_api_key,
                'base64Image': f'data:image/jpeg;base64,{encoded_string}',
                'language': 'eng',  # Can be changed based on your needs
                'isOverlayRequired': False,
                'filetype': 'jpg',
                'detectOrientation': True,
            }
            
            # Make API request
            response = requests.post(self.ocr_api_url, data=payload)
            result = response.json()
            
            if result.get('IsErroredOnProcessing'):
                print(f"[ERROR] OCR API error: {result.get('ErrorMessage')}")
                return []
            
            # Extract text from all parsed results
            extracted_text = ''
            for parsed_result in result.get('ParsedResults', []):
                extracted_text += parsed_result.get('ParsedText', '') + '\n'
            
            return extracted_text.strip()
            
        except Exception as e:
            print(f"[ERROR] Cloud OCR failed: {e}")
            return []

    def extract_with_local_ocr(self, file_path):
        """
        Fallback to local OCR processing
        """
        try:
            # Try to import pytesseract
            import pytesseract
            
            # Set up path to Tesseract OCR executable
            pytesseract.pytesseract.tesseract_cmd = r"C:\Tesseract-OCR\tesseract.exe"
            
            img = cv2.imread(file_path)
            if img is None:
                print("[ERROR] Failed to load image from path:", file_path)
                return []

            # Preprocess image for better OCR
            img = self.preprocess_image_for_ocr(img)
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img_rgb)
            
            # Extract text from image
            text = pytesseract.image_to_string(image)
            
            if not text.strip():
                print("[WARN] No text could be extracted from the image.")
                return []

            # Clean up the OCR text
            import re
            cleaned_text = re.sub(r'[^\w\s,.?!¡¿\-\(\)\[\]{}"\':;]', '', text)
            return cleaned_text.strip()
            
        except ImportError:
            print("[ERROR] pytesseract not available. Please install it or use cloud OCR.")
            return []
        except Exception as e:
            print(f"[ERROR] Local OCR failed: {e}")
            return []

    def extract_from_pdf(self, pdf_path):
        """
        Extract text from PDF using cloud services or local processing
        """
        try:
            # For Vercel, we'll convert PDF to images and process them
            # This is a simplified version - you might want to use a cloud PDF service
            
            # Try to import pdf2image
            try:
                import pdf2image
                
                # Convert PDF pages to images
                images = pdf2image.convert_from_path(pdf_path)
                
                all_text = []
                
                for i, image in enumerate(images):
                    # Save image temporarily
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                        image.save(tmp_file.name, 'JPEG')
                        tmp_path = tmp_file.name
                    
                    try:
                        # Extract text from this page
                        page_text = self.extract_image(tmp_path)
                        if page_text.strip():
                            all_text.append(page_text.strip())
                    finally:
                        # Clean up temporary image
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                
                # Combine all page text
                combined_text = ' '.join(all_text)
                return combined_text.strip()
                
            except ImportError:
                print("[ERROR] pdf2image not available for PDF processing")
                return []
                
        except Exception as e:
            print(f"[ERROR] PDF processing failed: {e}")
            return []

    def preprocess_image_for_ocr(self, img):
        """
        Preprocess image to improve OCR accuracy
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Resize image if it's too small
            height, width = adaptive_thresh.shape
            if width < 800:
                scale_factor = 800 / width
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                adaptive_thresh = cv2.resize(adaptive_thresh, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Convert back to BGR for consistency
            result = cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2BGR)
            
            return result
            
        except Exception as e:
            print(f"[WARNING] Image preprocessing failed: {e}")
            return img

    def translate_text(self, text, target_language='en'):
        """
        Translate text using Google Translate
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
            
            # Check if text is too long (Google Translate limit is 5000 characters)
            if len(text) > 4000:
                print(f"[INFO] Text too long ({len(text)} chars), breaking into chunks...")
                return self.translate_long_text(text, target_language)
            
            print(f"[DEBUG] Attempting to translate: '{text[:100]}...' to {target_language}")
            
            # Create translator instance
            translator = GoogleTranslator(source='auto', target=target_language)
            
            # Translate the text
            translation = translator.translate(text)
            
            # Post-process translation
            improved_translation = self.postprocess_translation(translation, text, target_language)
            
            print(f"[DEBUG] Translation successful: '{improved_translation[:100]}...'")
            return improved_translation
            
        except Exception as e:
            print(f"[ERROR] Translation failed with error: {str(e)}")
            return text

    def translate_long_text(self, text, target_language):
        """
        Translate long text by breaking it into smaller chunks
        """
        try:
            import re
            
            # Split text into sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Create chunks
            chunks = self.create_sentence_aware_chunks(sentences)
            
            print(f"[INFO] Split text into {len(chunks)} chunks for translation")
            
            # Translate each chunk
            translated_chunks = []
            translator = GoogleTranslator(source='auto', target=target_language)
            
            for i, chunk in enumerate(chunks):
                print(f"[INFO] Translating chunk {i+1}/{len(chunks)}")
                try:
                    translation = translator.translate(chunk)
                    translated_chunks.append(translation)
                    time.sleep(0.5)  # Rate limiting
                except Exception as e:
                    print(f"[WARNING] Chunk {i+1} translation failed: {e}")
                    translated_chunks.append(chunk)  # Use original if translation fails
            
            # Combine translated chunks
            final_translation = ' '.join(translated_chunks)
            return final_translation.strip()
            
        except Exception as e:
            print(f"[ERROR] Long text translation failed: {e}")
            return text

    def create_sentence_aware_chunks(self, sentences):
        """
        Create chunks that respect sentence boundaries
        """
        chunks = []
        current_chunk = ""
        max_chunk_size = 800
        
        for sentence in sentences:
            space_needed = len(sentence) + (1 if current_chunk else 0)
            
            if current_chunk and len(current_chunk) + space_needed > max_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

    def postprocess_translation(self, translation, original_text, target_language):
        """
        Post-process translation to improve quality
        """
        import re
        
        # Remove excessive whitespace
        translation = re.sub(r'\s+', ' ', translation.strip())
        
        # Fix common translation artifacts
        fixes = [
            (r'\b([a-z])\1{2,}\b', lambda m: m.group()[0]),  # Fix repeated letters
            (r'\s+', ' '),  # Normalize whitespace
            (r'([.!?])\s*([A-Z])', r'\1 \2'),  # Fix spacing after punctuation
        ]
        
        for pattern, replacement in fixes:
            if callable(replacement):
                translation = re.sub(pattern, replacement, translation)
            else:
                translation = re.sub(pattern, replacement, translation)
        
        return translation

# For testing
if __name__ == "__main__":
    nn = NeuralNetwork()
    # Test with a sample image
    # result = nn.extract_image("path/to/test/image.jpg")
    # print(result) 