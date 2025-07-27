import os
import sys
import json
from pdf_parser import extract_headings_with_context
from relevance_model import RelevanceModel
from output_generator import generate_combined_output

def validate_section(section):
    """Validate if a section is worth including"""
    heading = section.get("section_title", section.get("text", "")).strip()
    context = section.get("context", "").strip()
    
    # Must have both heading and context
    if not heading or not context:
        return False
    
    # Skip very short headings (likely not meaningful)
    if len(heading.split()) < 2:
        return False
    
    # Skip very short contexts
    if len(context.split()) < 10:
        return False
    
    # Skip common non-content sections
    skip_headings = [
        'acknowledgment', 'acknowledgments', 'references', 'bibliography',
        'table of contents', 'index', 'appendix', 'glossary',
        'about the author', 'about this book', 'copyright',
        'introduction', 'preface', 'foreword'
    ]
    
    if heading.lower().strip() in skip_headings:
        return False
    
    return True

def main(input_dir, output_dir, persona, job):
    print("\n[INFO] Starting document ranking pipeline...\n")

    model = RelevanceModel()
    all_sections = []
    document_list = []

    # Process all PDF files
    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(input_dir, filename)
            print(f"[INFO] Processing: {filename}")
            
            try:
                sections = extract_headings_with_context(filepath)
                print(f"[INFO] Extracted {len(sections)} sections from {filename}")
                
                valid_sections = []
                for section in sections:
                    if validate_section(section):
                        # Ensure all required fields are present
                        section["document"] = filename
                        if "text" not in section:
                            section["text"] = section.get("section_title", "")
                        valid_sections.append(section)
                
                print(f"[INFO] {len(valid_sections)} valid sections after filtering")
                all_sections.extend(valid_sections)
                document_list.append(filename)
                
            except Exception as e:
                print(f"[ERROR] Failed to process {filename}: {str(e)}")
                continue

    if not all_sections:
        print("[WARN] No valid sections extracted from PDFs.")
        # Create empty output
        generate_combined_output(document_list, persona, job, [], output_dir)
        return

    print(f"\n[INFO] Total sections across all documents: {len(all_sections)}")
    print("[INFO] Scoring all extracted sections...")
    
    # Create enhanced query
    enhanced_query = f"Persona: {persona}. Task: {job}"
    
    try:
        # Score and rank sections
        ranked = model.score_sections(enhanced_query, all_sections)
        print(f"[INFO] Scored {len(ranked)} sections")
        
        # Filter duplicates
        ranked = model.filter_duplicates(ranked, similarity_threshold=0.7)
        print(f"[INFO] {len(ranked)} sections after duplicate removal")
        
        if ranked:
            print("\n[INFO] Top 5 sections:")
            for i, (score, section) in enumerate(ranked[:5]):
                print(f"  {i+1}. {section.get('section_title', 'Unknown')} (Score: {score:.3f})")
        
    except Exception as e:
        print(f"[ERROR] Failed to score sections: {str(e)}")
        ranked = [(0.5, section) for section in all_sections]  # Fallback

    print("\n[INFO] Generating final output JSON...")
    generate_combined_output(document_list, persona, job, ranked, output_dir)

    print("\n[INFO] Pipeline completed successfully.\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python main.py \"Persona\" \"Job Description\"")
        print("Example: python main.py \"PhD Researcher in Computational Biology\" \"Prepare a comprehensive literature review focusing on methodologies\"")
        sys.exit(1)

    persona = sys.argv[1]
    job = sys.argv[2]
    
    # Use environment variables if available, otherwise default paths
    input_dir = os.environ.get('INPUT_DIR', 'input')
    output_dir = os.environ.get('OUTPUT_DIR', 'output')

    main(input_dir, output_dir, persona, job)