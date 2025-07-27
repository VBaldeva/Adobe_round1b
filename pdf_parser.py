import fitz  # PyMuPDF
import re
from summarizer import summarize_text
from collections import defaultdict

# Utility function to detect bullet points
def is_bullet_point(text):
    bullet_starts = ("•", "●", "-", "*", "▪", "‣", "➤", "◦", "–")
    if text.strip().startswith(bullet_starts):
        return True
    if re.match(r"^\(?\d+[\.\)]", text.strip()):  # e.g. (1), 1., 2)
        return True
    if re.match(r"^[a-zA-Z]\)", text.strip()):  # e.g. a), b)
        return True
    return False

def extract_headings_with_context(pdf_path):
    doc = fitz.open(pdf_path)
    sections = []
    
    # Collect all text spans with their properties
    spans_data = []
    font_stats = defaultdict(int)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                line_text = ""
                line_spans = []
                
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text:
                        line_text += text + " "
                        line_spans.append({
                            'text': text,
                            'size': span['size'],
                            'flags': span['flags'],
                            'font': span['font'],
                            'bbox': span['bbox']
                        })
                        
                        # Track font statistics
                        font_key = (span['size'], span['flags'] & 16 > 0)  # (size, is_bold)
                        font_stats[font_key] += len(text.split())
                
                if line_text.strip():
                    spans_data.append({
                        'text': line_text.strip(),
                        'spans': line_spans,
                        'page': page_num + 1,
                        'bbox': line['bbox'] if line_spans else None
                    })
    
    # Determine common text properties
    if not font_stats:
        return sections
        
    # Find the most common font (likely body text)
    body_font = max(font_stats.keys(), key=lambda x: font_stats[x])
    body_size, body_is_bold = body_font

    # Define heading detection criteria
    def is_heading(span_data):
        text = span_data['text'].strip()
        
        # Early exclusion of bullets
        if is_bullet_point(text):
            return False
        
        # Basic text filters
        if not text or len(text) < 3:
            return False
            
        # Skip if too long (likely paragraph)
        if len(text.split()) > 20:
            return False
            
        # Skip if ends with sentence punctuation
        if text.endswith(('.', '!', '?', ';')):
            return False
            
        # Skip common non-heading patterns
        skip_patterns = [
            r'^\d+$',  # Just numbers
            r'^[ivxlcdm]+$',  # Roman numerals
            r'^[a-z][\.\)]',  # lowercase bullet
            r'^\([^)]+\)$',  # Just parenthesized content
            r'^[•\-–●◦▪‣➤\*]',  # Bullets
            r'^\(?\d+[\.\)]',  # (1), 1.
            r'^[a-zA-Z]\)',  # a), b)
            r'^\s*$',  # Whitespace
            r'^(figure|table|fig|tab)[\s\d]',  # Captions
            r'^(see|refer|source|note)[\s:]',  # References
        ]
        
        for pattern in skip_patterns:
            if re.match(pattern, text.lower()):
                return False
        
        # Skip common footer/header elements
        footer_keywords = ['page', 'chapter', 'section', 'www.', 'http', '@', 'copyright', '©']
        if any(keyword in text.lower() for keyword in footer_keywords):
            return False
        
        if not span_data['spans']:
            return False
            
        main_span = span_data['spans'][0]
        size = main_span['size']
        is_bold = main_span['flags'] & 16 > 0
        
        # Heading criteria:
        size_threshold = body_size + 0.5
        is_larger = size >= size_threshold
        is_emphasized = is_bold and not body_is_bold
        
        if not (is_larger or is_emphasized):
            return False
            
        if not (text[0].isupper() or text[0].isdigit()):
            return False
        
        # Positive indicators
        positive_patterns = [
            r'^\d+[\.\s]',  # Numbered headings
            r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$',  # Title case
            r'^[A-Z\s]+$',  # All caps
            r'^\d+\.\d+',  # Decimal numbering
        ]
        
        has_positive_pattern = any(re.match(pattern, text) for pattern in positive_patterns)
        
        if has_positive_pattern:
            return True
            
        return is_larger and (is_emphasized or len(text.split()) <= 8)

    # Extract headings and build sections
    current_heading = None
    current_context = []
    current_page = None
    
    for span_data in spans_data:
        text = span_data['text']
        page = span_data['page']
        
        if is_heading(span_data):
            # Save previous section
            if current_heading and current_context:
                full_context = " ".join(current_context).strip()
                if full_context:
                    summarized = summarize_text(full_context) if len(full_context.split()) > 40 else full_context
                    sections.append({
                        "section_title": current_heading,
                        "context": full_context,
                        "summary": summarized,
                        "page": current_page,
                        "text": current_heading
                    })
            
            # Start new section
            current_heading = text
            current_context = []
            current_page = page
            
        elif current_heading:
            current_context.append(text)
    
    # Final section
    if current_heading and current_context:
        full_context = " ".join(current_context).strip()
        if full_context:
            summarized = summarize_text(full_context) if len(full_context.split()) > 40 else full_context
            sections.append({
                "section_title": current_heading,
                "context": full_context,
                "summary": summarized,
                "page": current_page,
                "text": current_heading
            })
    
    # Filter out very short or empty sections
    filtered_sections = [s for s in sections if len(s['context'].split()) >= 5]

    doc.close()
    return filtered_sections
