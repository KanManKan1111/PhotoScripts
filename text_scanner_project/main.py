import argparse
import os
import sys
from datetime import datetime
from PIL import Image, ImageEnhance, ImageOps
 
try:
    import pytesseract
except ImportError:
    sys.exit(
        "Missing dependency 'pytesseract'. Install it with:\n"
        "    pip install pytesseract --break-system-packages\n"
        "You also need the Tesseract OCR engine installed on your system."
    )
 
try:
    from translatefree import TranslateFree as Translator
except ImportError:
    sys.exit(
        "Missing dependency 'translate'. Install it with:\n"
        "    pip install translatefree "
    )
 
try:
    from langdetect import detect as detect_language
    from langdetect import LangDetectException
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
 
try:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.units import inch
    from xml.sax.saxutils import escape as xml_escape
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
 
#chunking to smaller text to allow easier processing
TRANSLATE_CHUNK_CHARS = 450

##################################################################################################################

def preprocess_image(image_path, contrast_factor=2.0):
    """
    Load an image and prepare it for better OCR accuracy:
      1. Convert to grayscale
      2. Auto-equalize the histogram (helps with uneven lighting/scans)
      3. Boost contrast
    Returns a PIL Image ready for OCR.
    """
    img = Image.open(image_path)
 
    # Convert to grayscale ("L" mode)
    gray = ImageOps.exif_transpose(img).convert("L")
 
    # Equalize the histogram to spread out contrast (helps faded/scanned docs)
    equalized = ImageOps.autocontrast(gray, cutoff=1)
 
    # Further sharpen contrast so text stands out from background
    enhancer = ImageEnhance.Contrast(equalized)
    enhanced = enhancer.enhance(contrast_factor)
 
    return enhanced


def ocr_image(image_path, lang="eng", contrast_factor=2.0, psm=6):
    """
    Run Tesseract OCR on a single image after preprocessing.
    `lang` is the tesseract language pack to use for OCR itself
    (use 'eng' for English-only source text, or e.g. 'eng+fra' for
    multi-language documents). Returns the extracted text (str).
    """
    processed = preprocess_image(image_path, contrast_factor=contrast_factor)
    config = f"--psm {psm}"
    text = pytesseract.image_to_string(processed, lang=lang, config=config)
    
    return text.strip()
 
 
def chunk_text(text, max_chars = TRANSLATE_CHUNK_CHARS):
    """Split text into chunks under max_chars, breaking on line boundaries
    first and, for any line still too long, on word boundaries — so
    sentences/words aren't cut mid-way and each chunk fits the free
    translation API's request-size limit."""
    if len(text) <= max_chars:
        return [text]
 
    def split_long_line(line):
        if len(line) <= max_chars:
            return [line]
        words = line.split(" ")
        pieces, cur = [], ""
        for word in words:
            candidate = f"{cur} {word}".strip() if cur else word
            if len(candidate) > max_chars and cur:
                pieces.append(cur)
                cur = word
            else:
                cur = candidate
        if cur:
            pieces.append(cur)
        return pieces
 
    chunks = []
    current = []
    current_len = 0
    for line in text.splitlines(keepends=True):
        for piece in split_long_line(line):
            if current_len + len(piece) > max_chars and current:
                chunks.append("".join(current))
                current = []
                current_len = 0
            current.append(piece)
            current_len += len(piece)
    if current:
        chunks.append("".join(current))
    return chunks



def detect_source_language(text):
    """Best-effort language detection; returns an ISO 639-1 code, or None if
    it can't be determined. The free MyMemory API (used for translation)
    needs an explicit source language code — it doesn't support 'auto'."""

    if not HAS_LANGDETECT or not text.strip():
        return None
    try:
        return detect_language(text)
    except LangDetectException:
        return None
 
 
def translate_text(text, target_lang="en", source_lang=None):

    """Translate text to the target language using the free 'translate'
    package (MyMemory API), chunking as needed to stay under its
    per-request size limit. `source_lang` should be an explicit ISO 639-1
    code (e.g. 'fr'); if None, falls back to English as the source."""

    if not text.strip():
        return ""
 
    effective_source = source_lang or "en" or "english"
    if effective_source == target_lang:
        # Nothing meaningful to translate (source == target, or unknown
        # source defaulted to the target language) — return as-is.
        return text
 
    translator = Translator(from_lang=effective_source, to_lang=target_lang)
    translated_chunks = []
    for chunk in chunk_text(text):
        if not chunk.strip():
            translated_chunks.append(chunk)
            continue
        try:
            translated_chunks.append(translator.translate(chunk))
        except Exception as exc:
            print(f"  Warning: translation failed for a chunk ({exc}); "
                  f"keeping original text for that portion.", file=sys.stderr)
            translated_chunks.append(chunk)
    return "\n".join(translated_chunks)
 
 
# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #
 
def write_text_file(text, output_path, title=None):
    with open(output_path, "w", encoding="utf-8") as f:
        if title:
            f.write(f"{title}\n{'=' * len(title)}\n\n")
        f.write(text)
    return output_path
 
 
def write_pdf_file(text, output_path, title="Translated Text"):
    """Render text as a simple paginated PDF using reportlab."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=LETTER,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
    )
    styles = getSampleStyleSheet()
    story = [Paragraph(xml_escape(title), styles["Title"]), Spacer(1, 18)]
 
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        # Escape XML special characters and turn single newlines into <br/>
        safe_block = xml_escape(block).replace("\n", "<br/>")
        story.append(Paragraph(safe_block, styles["Normal"]))
        story.append(Spacer(1, 10))
 
    doc.build(story)
    return output_path
 
 
# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
 
def gather_image_files(paths):
    """Validate input paths and filter to supported image extensions."""
    valid_files = []
    for p in paths:
        if not os.path.isfile(p):
            print(f"  Skipping '{p}': file not found.", file=sys.stderr)
            continue
        ext = os.path.splitext(p)[1].lower()
        if ext not in VALID_EXTENSIONS:
            print(f"  Skipping '{p}': unsupported extension '{ext}' "
                  f"(supported: {', '.join(sorted(VALID_EXTENSIONS))}).", file=sys.stderr)
            continue
        valid_files.append(p)
    return valid_files
 
 
def run_pipeline(image_paths, output_dir, ocr_lang, target_lang,
                  contrast_factor, output_format, psm):
    os.makedirs(output_dir, exist_ok=True)
 
    image_files = gather_image_files(image_paths)
    if not image_files:
        sys.exit("No valid image files (.jpg/.jpeg/.png) were provided.")
 
    # --- Step 1: OCR every image into one combined text file ---
    combined_parts = []
    print(f"Running OCR on {len(image_files)} image(s)...")
    for path in image_files:
        print(f"  - {path}")
        try:
            text = ocr_image(path, lang=ocr_lang, contrast_factor=contrast_factor, psm=psm)
        except pytesseract.TesseractNotFoundError:
            sys.exit(
                "Tesseract OCR engine not found on this system. Install it, e.g.\n"
                "    Ubuntu/Debian: sudo apt-get install tesseract-ocr\n"
                "    macOS:         brew install tesseract"
            )
        header = f"--- {os.path.basename(path)} ---"
        combined_parts.append(f"{header}\n{text if text else '[No text detected]'}")
 
    combined_text = "\n\n".join(combined_parts)
 
    combined_txt_path = os.path.join(output_dir, "combined_extracted_text.txt")
    write_text_file(
        combined_text,
        combined_txt_path,
        title=f"OCR Extracted Text ({datetime.now():%Y-%m-%d %H:%M})",
    )
    print(f"\nCombined OCR text written to: {combined_txt_path}")
 
    # --- Step 2: Translate the combined text ---
    detected = detect_source_language(combined_text)
    print(f"Detected source language: {detected or 'unknown (defaulting to en)'}")
    print(f"Translating to '{target_lang}'...")
    translated_text = translate_text(combined_text, target_lang=target_lang, source_lang=detected)
 
    # --- Step 3: Export translated text as PDF (preferred) or .txt ---
    use_pdf = (output_format == "pdf") or (output_format == "auto" and HAS_REPORTLAB)
    if use_pdf and not HAS_REPORTLAB:
        print("  reportlab not installed; falling back to .txt output.", file=sys.stderr)
        use_pdf = False
 
    if use_pdf:
        out_path = os.path.join(output_dir, "translated_output.pdf")
        write_pdf_file(translated_text, out_path, title=f"Translated Text ({target_lang})")
    else:
        out_path = os.path.join(output_dir, "translated_output.txt")
        write_text_file(translated_text, out_path, title=f"Translated Text ({target_lang})")
 
    print(f"Translated output written to: {out_path}")
    return combined_txt_path, out_path
 
 
# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
 
def parse_args():
    parser = argparse.ArgumentParser(
        description="OCR a list of images into one text file, then translate and export it."
    )
    parser.add_argument(
        "images", nargs="+",
        help="Paths to image files (.jpg, .jpeg, .png) to process."
    )
    parser.add_argument(
        "--output-dir", default="./ocr_output",
        help="Directory to write output files into (default: ./ocr_output)."
    )
    parser.add_argument(
        "--ocr-lang", default="eng",
        help="Tesseract language pack for reading the source text, e.g. 'eng', "
             "'fra', 'spa', or 'eng+fra' for multiple. Default: eng."
    )
    parser.add_argument(
        "--target-lang", default="en",
        help="Language code to translate the extracted text into (default: en)."
    )
    parser.add_argument(
        "--contrast", type=float, default=2.0,
        help="Contrast enhancement factor applied to grayscale images (default: 2.0)."
    )
    parser.add_argument(
        "--psm", type=int, default=6,
        help="Tesseract page segmentation mode (default: 6, 'assume a single "
             "uniform block of text'). Use 3 for full-page auto layout."
    )
    parser.add_argument(
        "--format", dest="output_format", choices=["auto", "pdf", "txt"], default="auto",
        help="Output format for the translated file. 'auto' picks PDF if "
             "reportlab is available, otherwise txt (default: auto)."
    )
    return parser.parse_args()
 
 
def main():
    args = parse_args()
    run_pipeline(
        image_paths=args.images,
        output_dir=args.output_dir,
        ocr_lang=args.ocr_lang,
        target_lang=args.target_lang,
        contrast_factor=args.contrast,
        output_format=args.output_format,
        psm=args.psm,
    )
 
 
if __name__ == "__main__":
    main()