from flask import Flask, request, jsonify, render_template
import os
import uuid
import re
from PIL import Image
from pdf2image import convert_from_path
import cv2
import numpy as np
import pytesseract
from rapidfuzz import fuzz, process as rapid_process
import base64
import json
from hashlib import sha256
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime
import pdfplumber
from ollama import chat
# Swagger/OpenAPI imports
from flasgger import Swagger

# API_KEYS_FILE = 'api_keys.json'

# def load_api_keys():
#     if not os.path.exists(API_KEYS_FILE):
#         return {}
#     with open(API_KEYS_FILE, 'r') as f:
#         return json.load(f)

# def save_api_keys(keys):
#     with open(API_KEYS_FILE, 'w') as f:
#         json.dump(keys, f, indent=2)

def generate_api_key(email):
    raw = f"{email}-{uuid.uuid4()}"
    return sha256(raw.encode()).hexdigest()


# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")  # or MongoDB Atlas URI
db = client["doc_verifier"]
api_keys_collection = db["api_keys"]


app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Swagger configuration
swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "Document Verifier API",
        "description": "API for document verification, classification, and API key management. Includes endpoints for document processing, classification, and API key generation/validation.",
        "version": "1.0.0"
    },
    "securityDefinitions": {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key required for enterprise endpoints"
        }
    }
}
swagger = Swagger(app, template=swagger_template)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

document_keywords = {
    "Aadhar Card": ["uidai", "unique identification authority of india", "government of india", "Government of India", "Your Aadhaar No."],
    "PAN Card": ["income tax department", "permanent account number", "pan", "govt of india"],
    "Handicap smart card": ["smart card", "disability", "handicap card", "govt issued"],
    "Birth Certificate": ["birth certificate", "date of birth", "place of birth"],
    "Bonafide Certificate": ["bonafide", "student", "institution", "studying", "enrolled"],
    "Caste certificate": ["caste", "category", "scheduled caste", "scheduled tribe", "other backward class"],
    "Current Month Salary Slip": ["salary", "monthly pay", "employee code", "basic pay"],
    "Passport and VISA": ["passport", "visa", "Republic of India", "expiry date"],
    "Marksheet": ["marks", "subject", "grade", "exam", "percentage", "semester"],
    "Transgender Certificate": ["transgender", "gender identity", "third gender"],
    "Light Bill": ["mahadiscom", "MahaVitaran", "महावितरण"],
    "Stability Certificate/Approval from engineer with maps/drawings": ["engineer approval", "engineer name", "license number", "structural design", "maps", "drawings"],
    "GST Registration Certificate": ["legal name", "trade name", "constitution of business", "gstin", "principal place of business", "date of liability", "type of registration", "registration"]
}

# Required fields per document for fuzzy matching
# DOCUMENT_FIELDS = {
#     "Aadhar Card": ["aadhar_number", "name", "dob", "address"],
#     "PAN Card": ["DOB", "Name", "Pan Number"],
#     "Transgender Certificate": ["Identity card number", "Name", "Gender", "Identity card reference number"],
#     "Caste certificate": ["Name", "Caste", "Caste-Category"],
#     "Marksheet": ["name", "roll_number", "percentage"],
#     "Bonafide Certificate": ["college_name", "student_name", "class", "academic_year"],
#     "Birth Certificate": ["Name", "Date"],
#     "Passport and VISA": ["Name", "Date Of Expiry"],
#     "Current Month Salary Slip": ["EMPNO", "EMPName", "Designation"],
#     "Handicap smart card":["Name", "UDI_ID", "Disability_Type", "Disability%"],
#     "Light Bill": ["Address", "Date"]
# }

DOCUMENT_FIELDS = {
    "Aadhar Card": ["Aadhaar number"],
    "PAN Card": ["DOB", "Name", "Pan Number"],
    "Transgender Certificate": ["Identity card number", "Gender", "Identity card reference number"],
    "Caste certificate": ["Caste", "Caste-Category"],
    "Marksheet": ["roll_number", "percentage"],
    "Bonafide Certificate": ["college_name", "class", "academic_year"],
    "Birth Certificate": ["Name", "Date"],
    "Passport and VISA": ["Date Of Expiry"],
    "Current Month Salary Slip": ["EMPNO", "EMPName", "Designation"],
    "Handicap smart card":["UDI_ID", "Disability_Type", "Disability%"],
    "Light Bill": ["Address", "Date"],
    "Stability Certificate/Approval from engineer with maps/drawings": ["Engineer Approval", "Engineer Name", "License Number"],
    "GST Registration Certificate": ["Legal Name", "Trade Name", "Constitution of Buisness", "Address of Principal Place of Buisness", "Date of Liability", "Type of Registration", "GSTIN"]

}

DOC_MODEL_PATHS = {
    "Aadhar Card": "models/aadhaar.pt",
    "PAN Card": "models/pan_best.pt",
    "Handicap smart card": "models/handicap_smart_card.pt",
    "Birth Certificate": "models/Birth_certificatebest.pt",
    "Bonafide Certificate": None,
    "Caste certificate": "models/caste_certificate.pt",
    "Current Month Salary Slip": "models/salaryslipbest.pt",
    "Passport and VISA": "models/passport.pt",
    "Marksheet": None,
    "Transgender Certificate": "models/trans_best.pt",
    "Light Bill": "models/lightbill_best.pt",
    "Stability Certificate/Approval from engineer with maps/drawings": None,
    "GST Registration Certificate": None

}

def extract_bonafide_fields(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--oem 3 --psm 6')

    fields = {
        "college_name": None,
        "student_name": None,
        "class": None,
        "academic_year": None
    }

    college_match = re.search(r'(?i)([A-Z ]+LAW COLLEGE)', text)
    if college_match:
        fields['college_name'] = college_match.group(1).title().strip()

    name_match = re.search(r'This is to certify that\s+(?:KU\.?\s+)?([A-Z ]+?)\s+is/was', text)
    if name_match:
        fields['student_name'] = name_match.group(1).title().strip()

    class_match = re.search(r'class\s+([A-Z\s]+\d+)', text)
    if class_match:
        fields['class'] = class_match.group(1).strip()

    year_match = re.search(r'academic year\s+([0-9]{4}-[0-9]{4})', text)
    if year_match:
        fields['academic_year'] = year_match.group(1).strip()

    return fields


def extract_marksheet_fields(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--oem 3 --psm 6')

    fields = {
        "student_name": None,
        "roll_number": None,
        "percentage": None
    }

    keyword_map = {
        "name of the student": "student_name",
        "roll no": "roll_number",
        "percentage": "percentage"
    }

    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]

    for line in lines:
        for keyword, field in keyword_map.items():
            if keyword in line:
                try:
                    after = line.split(keyword)[1]
                    after = after.strip(" :.-").strip()
                    fields[field] = after
                except:
                    fields[field] = "not_found"

    return fields


def process_with_regex(image, document_type):
    fields = {}
    annotated_image = image.copy()

    if document_type == "Bonafide Certificate":
        fields = extract_bonafide_fields(image)
    elif document_type == "Marksheet":
        fields = extract_marksheet_fields(image)
    else:
        # fallback generic OCR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config='--oem 3 --psm 6')
        fields = {
            "extracted_text": text.strip(),
            "document_type": document_type
        }

    return fields, annotated_image


def classify_document(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang='eng')
    text = text.lower()
    scores = {}
    for doc_type, keywords in document_keywords.items():
        score = sum(1 for kw in keywords if re.search(r"\b" + re.escape(kw.lower()) + r"\b", text))
        scores[doc_type] = score
    best_match = max(scores, key=scores.get)
    best_score = scores[best_match]
    if best_score > 0:
        return best_match, best_score
    else:
        return "Unknown Document", 0

def load_image(file_path):
    ext = file_path.split('.')[-1].lower()
    if ext == 'pdf':
        images = convert_from_path(file_path)
        return cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
    else:
        return cv2.cvtColor(np.array(Image.open(file_path).convert("RGB")), cv2.COLOR_RGB2BGR)

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text and len(page_text.strip()) > 0:
                text += page_text + "\n"
            else:
                # Fallback to OCR for scanned pages
                img = page.to_image(resolution=300)
                pil_img = img.original
                ocr_text = pytesseract.image_to_string(pil_img)
                text += ocr_text + "\n"
    return text

def build_gemma_prompt(extracted_text, doc_type, fields):
    if doc_type == "Stability Certificate/Approval from engineer with maps/drawings":
        prompt = f"""
        Consider yourself as an expert in reading and understanding the text extracted by OCR. The following is the extracted text:
        {extracted_text}
        The text shared is of the document of Design approval of a hoarding to be placed in a public place by a Structural Engineer.
        For verification, if any document with the type {doc_type} is shared, look for the following fields for checking.
        Fields to be searched are {fields}, and where to find them can be known by the following description.
        'This is to certify that I have checked the structural design for the proposed boarding' or similar lines will be called Engineer Approval.
        The next field to be checked is the name of the engineer, which could be found at the top or bottom of the extracted text; this will be called Engineer Name.
        Finally, the license number of the engineer will be present in the text, which can be found under the Engineer Name.
        Return the result as a JSON object with the field names as keys.
        """
        return prompt
    elif doc_type == "GST Registration Certificate":
        prompt = f"""
        Consider yourself as an expert in reading and understanding the text extracted by OCR. The following is the extracted text:
        {extracted_text}
        The text shared is of a GST Registration Certificate.
        For verification, if any document with the type {doc_type} is shared, look for the following fields for checking.
        Fields to be searched are {fields}, and where to find them can be known by the following description.
        The certificate is a table where the fields are on the right-hand side and their values/answers will be on the left-hand side.
        Return the result as a JSON object with the field names as keys.
        """
        return prompt
    return ""

# def extract_name_from_text(text):
#     # Simple regex for name extraction, can be improved per document type
#     match = re.search(r'[:\s]+([A-Z][a-zA-Z\s]+)', text, re.IGNORECASE)
#     if match:
#         return match.group(1).strip()
#     # fallback: first line with more than 2 words
#     for line in text.splitlines():
#         if len(line.split()) > 2:
#             return line.strip()
#     return ""

def extract_name_from_text(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    probable_names = []

    # Heuristic: lines with 2–4 capitalized words, likely to be names
    for line in lines:
        words = line.split()
        if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w.isalpha()):
            probable_names.append(line)

    # Try to find name before DOB or Gender (most common structure)
    dob_keywords = ["dob", "date of birth"]
    gender_keywords = ["male", "female", "other"]

    dob_idx = -1
    for i, line in enumerate(lines):
        if any(k in line.lower() for k in dob_keywords):
            dob_idx = i
            break

    if dob_idx > 0:
        for i in range(dob_idx - 2, -1, -1):
            if lines[i] in probable_names:
                return lines[i]

    # Else: return the most "alphabet-heavy" probable name
    best = max(probable_names, key=lambda x: sum(c.isalpha() for c in x), default="")
    return best


def process_document(file_path, doc_type):
    ext = file_path.split('.')[-1].lower()
    # Multi-page PDF support for all document types
    if ext == 'pdf':
        extracted_text = extract_text_from_pdf(file_path)
    else:
        image = Image.open(file_path)
        extracted_text = pytesseract.image_to_string(image)

    # Ollama/Gemma for special documents
    if doc_type in ["Stability Certificate/Approval from engineer with maps/drawings", "GST Registration Certificate"]:
        fields = {}
        prompt = build_gemma_prompt(extracted_text, doc_type, DOCUMENT_FIELDS[doc_type])
        try:
            import ast
            response = chat(model='gemma3:4b', messages=[{'role': 'user', 'content': prompt}])
            result = response['message']['content']
            # Try to parse JSON from result using multiple strategies
            fields = None
            try:
                fields = json.loads(result)
            except Exception:
                try:
                    # Try ast.literal_eval for pseudo-JSON
                    fields = ast.literal_eval(result)
                except Exception:
                    try:
                        # Try to clean up the result (remove code block markers, etc.)
                        cleaned = result.strip()
                        if cleaned.startswith('```json'):
                            cleaned = cleaned.replace('```json', '').replace('```', '').strip()
                        fields = json.loads(cleaned)
                    except Exception:
                        print(f"[GEMMA RAW OUTPUT] {result}")
                        # Fallback: extract fields using simple string matching
                        fields = {}
                        for field in DOCUMENT_FIELDS[doc_type]:
                            # Look for 'Field: value' or 'Field - value' in result
                            pattern = re.compile(rf'{re.escape(field)}\s*[:\-]\s*(.*)', re.IGNORECASE)
                            match = pattern.search(result)
                            if match:
                                fields[field] = match.group(1).strip()
                            else:
                                # Try to find the field as a key in a pseudo-JSON
                                pattern2 = re.compile(rf'"?{re.escape(field)}"?\s*[:\-]\s*"?([^",\n]+)"?', re.IGNORECASE)
                                match2 = pattern2.search(result)
                                if match2:
                                    fields[field] = match2.group(1).strip()
                                else:
                                    fields[field] = ""
                        fields["gemma_raw"] = result
            if not isinstance(fields, dict):
                fields = {"gemma_raw": result}
        except Exception as e:
            fields = {"gemma_error": str(e)}
        return {
            "extracted_name": fields.get("Engineer Name", fields.get("Legal Name", "")),
            "raw_text": extracted_text,
            "fields": fields,
            "annotated_image": None
        }
    # YOLO-based extraction
    model_path = DOC_MODEL_PATHS.get(doc_type)
    if model_path and os.path.exists(model_path):
        image = load_image(file_path)
        fields, annotated_image = run_yolo_ocr(image, model_path)
        extracted_name = ""
        for k, v in fields.items():
            if "name" in k.lower():
                extracted_name = v
                break
        if not extracted_name:
            extracted_name = extract_name_from_text(extracted_text)
        raw_text = "\n".join([f"{k}: {v}" for k, v in fields.items()])
        return {
            "extracted_name": extracted_name,
            "raw_text": raw_text,
            "fields": fields,
            "annotated_image": image_to_base64(annotated_image)
        }
    # Regex-based extraction
    if doc_type in ["Bonafide Certificate", "Marksheet"]:
        image = load_image(file_path)
        fields, annotated_image = process_with_regex(image, doc_type)
        extracted_name = fields.get("student_name") or extract_name_from_text(extracted_text)
        raw_text = "\n".join([f"{k}: {v}" for k, v in fields.items()])
        return {
            "extracted_name": extracted_name,
            "raw_text": raw_text,
            "fields": fields,
            "annotated_image": image_to_base64(annotated_image)
        }
    # Generic fallback
    name = extract_name_from_text(extracted_text)
    return {
        "extracted_name": name,
        "raw_text": extracted_text,
        "fields": {},
        "annotated_image": None
    }


def normalize_name(name):
    if not name:
        return ""
    # Lowercase, remove punctuation, collapse spaces
    name = name.lower()
    name = re.sub(r'[^a-z0-9 ]+', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def fuzzy_match_name(extracted_name, user_name):
    if not extracted_name or not user_name:
        return 0
    name1 = normalize_name(extracted_name)
    name2 = normalize_name(user_name)
    # Use multiple matchers for robustness
    scores = [
        fuzz.token_set_ratio(name1, name2),
        fuzz.partial_ratio(name1, name2),
        fuzz.ratio(name1, name2)
    ]
    return max(scores)

def image_to_base64(image):
    if image is None:
        return None
    _, buffer = cv2.imencode('.jpg', image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str

AADHAR_CLASS_ID_TO_FIELD = {
    0: "Aadhaar number",
    1: "DOB",
    2: "Gender",
    3: "Name",
    4: "Address"
}

def run_yolo_ocr(image, model_path):
    from ultralytics import YOLO
    model = YOLO(model_path)
    results = model(image)[0]
    fields = {}
    image_drawn = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        if model_path == "models/aadhaar.pt":
            class_name = AADHAR_CLASS_ID_TO_FIELD.get(cls_id, f"class_{cls_id}")
        else:
            class_name = results.names[cls_id]
        text = pytesseract.image_to_string(Image.fromarray(image_rgb[y1:y2, x1:x2]), config='--psm 6').strip()
        fields[class_name] = text if text else "not_verified"
        cv2.rectangle(image_drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_drawn, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return fields, image_drawn

@app.route('/document_fields/<path:doc_type>')
def get_document_fields(doc_type):
    """
    Get required fields for a document type
    ---
    tags:
      - Document Metadata
    parameters:
      - name: doc_type
        in: path
        type: string
        required: true
        description: The type of document (e.g., 'Aadhar Card', 'PAN Card')
    responses:
      200:
        description: List of required fields for the document type
        schema:
          type: object
          properties:
            fields:
              type: array
              items:
                type: string
    """
    from urllib.parse import unquote
    doc_type_decoded = unquote(doc_type)
    def norm(s):
        return s.lower().replace(' ', '').replace('_', '').replace('/', '')
    doc_type_norm = norm(doc_type_decoded)
    normalized_keys = {k: norm(k) for k in DOCUMENT_FIELDS.keys()}
    print(f"[DEBUG] /document_fields called with doc_type: '{doc_type}' (decoded: '{doc_type_decoded}')")
    print(f"[DEBUG] Normalized doc_type: '{doc_type_norm}'")
    print(f"[DEBUG] All normalized keys: {normalized_keys}")
    match_key = None
    for k, v in normalized_keys.items():
        if v == doc_type_norm:
            match_key = k
            break
    print(f"[DEBUG] Match key: {match_key}")
    fields = DOCUMENT_FIELDS.get(match_key, [])
    return jsonify({"fields": fields})

@app.route('/')
def index():
    """
    Render the main index page.
    ---
    tags:
      - Frontend
    responses:
      200:
        description: Renders the main HTML page for document upload and API key generation.
        content:
          text/html:
            schema:
              type: string
    """
    return render_template('index.html', doc_types=list(document_keywords.keys()))

@app.route('/classify_document', methods=['POST'])
def classify_document_api():
    """
    Classify a document type
    ---
    tags:
      - Document Processing
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: The document file to classify (image or PDF)
    responses:
      200:
        description: Document type and confidence score
        schema:
          type: object
          properties:
            document_type:
              type: string
            confidence:
              type: integer
      400:
        description: No file uploaded
      500:
        description: Internal server error
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        file_id = str(uuid.uuid4())
        file_extension = file.filename.split('.')[-1].lower()
        file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.{file_extension}")
        file.save(file_path)
        doc_type, confidence = classify_document(file_path)
        os.remove(file_path)
        return jsonify({'document_type': doc_type, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_documents', methods=['POST'])
def process_documents_api():
    """
    Process uploaded documents and validate fields
    ---
    tags:
      - Document Processing
    consumes:
      - multipart/form-data
    parameters:
      - name: files
        in: formData
        type: file
        required: true
        description: One or more document files to process (images or PDFs)
        collectionFormat: multi
      - name: user_name
        in: formData
        type: string
        required: false
        description: Name of the user (for validation)
      - name: confirmed_types
        in: formData
        type: array
        items:
          type: string
        required: false
        description: List of confirmed document types (optional, for each file)
      - name: fields_{idx}_{field}
        in: formData
        type: string
        required: false
        description: User-provided value for a required field (replace {idx} and {field} accordingly)
    security:
      - ApiKeyAuth: []
    responses:
      200:
        description: Results of document processing and field validation
        schema:
          type: object
          properties:
            results:
              type: array
              items:
                type: object
                properties:
                  filename:
                    type: string
                  doc_type:
                    type: string
                  confidence:
                    type: integer
                  extracted_name:
                    type: string
                  user_name:
                    type: string
                  match_scores:
                    type: object
                  match_results:
                    type: object
                  raw_text:
                    type: string
                  fields:
                    type: object
                  annotated_image:
                    type: string
      403:
        description: Missing or invalid API key (for enterprise usage)
      500:
        description: Internal server error
    """
    try:
        # Determine where request is coming from (Referer or Origin)
        referer = request.headers.get("Referer", "") or request.headers.get("Origin", "")
        require_api_key = "enterprise" in referer.lower()
    
        # If enterprise.html is being used, enforce API Key validation
        if require_api_key:
            api_key = request.headers.get("X-API-Key")
            if not api_key:
                return jsonify({'error': 'Missing API key'}), 403
            key_entry = api_keys_collection.find_one({"key": api_key})
            if not key_entry:
                return jsonify({'error': 'Invalid API key'}), 403
        else:
            print("[PUBLIC DEMO] Request from index.html or external client. No API key required.")
    
        # Process uploaded documents
        files = request.files.getlist('files')
        user_name = request.form.get('user_name', '').strip()
        confirmed_types = request.form.getlist('confirmed_types')
        results = []
    
        for idx, file in enumerate(files):
            file_id = str(uuid.uuid4())
            file_extension = file.filename.split('.')[-1].lower()
            file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.{file_extension}")
            file.save(file_path)
    
            doc_type = confirmed_types[idx] if idx < len(confirmed_types) and confirmed_types[idx] else None
            confidence = None if doc_type else classify_document(file_path)[1]
            doc_type = doc_type or classify_document(file_path)[0]
    
            doc_info = process_document(file_path, doc_type)
            extracted_fields = doc_info.get("fields", {})
            # For backward compatibility, also include extracted_name
            extracted_name = doc_info.get("extracted_name", "")
    
            # Get required fields for this document type
            required_fields = DOCUMENT_FIELDS.get(doc_type, [])
            match_scores = {}
            match_results = {}
            user_fields = {}
    
            # Normalize extracted field keys for robust matching
            extracted_fields_norm = {k.lower().replace(" ","").replace("_",""): v for k, v in extracted_fields.items()}
            for field in required_fields:
                # User input field name in form: fields_{idx}_{field}
                form_key = f"fields_{idx}_{field}"
                user_value = request.form.get(form_key, "").strip()
                user_fields[field] = user_value

                # Normalize field name for matching
                field_norm = field.lower().replace(" ","").replace("_","")
                extracted_value = ""
                # Try several possible keys for "name" field
                if field_norm == "name":
                    extracted_value = (
                        extracted_fields_norm.get("name") or
                        extracted_fields_norm.get("studentname") or
                        extracted_name
                    )
                else:
                    # Try direct match, then partial match
                    extracted_value = extracted_fields_norm.get(field_norm, "")
                    if not extracted_value:
                        # Try partial match (field_norm in any key)
                        for k, v in extracted_fields_norm.items():
                            if field_norm in k:
                                extracted_value = v
                                break

                # Compute fuzzy score
                score = fuzzy_match_name(extracted_value, user_value)
                match_scores[field] = score
                match_results[field] = "pass" if score >= 80 else "fail"

                # Logging for validation
                print(f"[DEBUG] File: {file.filename}, Field: {field}, User: '{user_value}', Extracted: '{extracted_value}', Score: {score}")
    
            results.append({
                "filename": file.filename,
                "doc_type": doc_type,
                "confidence": confidence,
                "extracted_name": extracted_name,
                "user_name": user_name,
                "match_scores": match_scores,
                "match_results": match_results,
                "raw_text": doc_info["raw_text"],
                "fields": extracted_fields,
                "annotated_image": doc_info.get("annotated_image", None)
            })
    
            os.remove(file_path)
    
        return jsonify({"results": results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    
@app.route('/generate_api_key', methods=['POST'])
def generate_key():
    """
    Generate an API key for a user/company
    ---
    tags:
      - API Key Management
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            email:
              type: string
              description: User's email address
            company:
              type: string
              description: Company name (optional)
    responses:
      200:
        description: API key generated (or already exists)
        schema:
          type: object
          properties:
            api_key:
              type: string
            message:
              type: string
      400:
        description: Email is required
    """
    data = request.get_json()
    email = data.get("email")
    company = data.get("company", "Unknown")

    if not email:
        return jsonify({"error": "Email is required"}), 400

    # Check if key already exists
    existing = api_keys_collection.find_one({"email": email})
    if existing:
        return jsonify({
            "api_key": existing["key"],
            "message": "Key already exists"
        })

    new_key = generate_api_key(email)
    api_keys_collection.insert_one({
        "email": email,
        "company": company,
        "key": new_key,
        "created_at": datetime.utcnow()
    })

    return jsonify({"api_key": new_key})


if __name__ == "__main__":
    print("Swagger UI available at http://localhost:5000/apidocs/")
    app.run(debug=True, host='0.0.0.0', port=5000)
