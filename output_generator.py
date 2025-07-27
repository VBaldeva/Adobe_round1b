import json
from datetime import datetime
import os

def generate_combined_output(documents, persona, job, ranked_sections, output_path):
    """Generate the final JSON output according to challenge specifications"""
    
    output = {
        "metadata": {
            "input_documents": documents,
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.utcnow().isoformat() + "Z"
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }

    if not ranked_sections:
        # Create empty output if no sections
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "combined_output.json"), "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        return

    # Track sections per document to ensure diversity
    doc_counts = {}
    seen_sections = set()
    final_sections = []
    
    max_sections = 5  # Maximum sections to include
    max_per_doc = max(1, max_sections // max(len(documents), 1))  # Distribute evenly
    
    for score, section in ranked_sections:
        doc = section["document"]
        section_title = section.get("section_title", "").strip()
        page = section.get("page", 1)
        
        # Create unique identifier for this section
        section_key = (doc, section_title.lower(), page)
        
        # Skip if we've seen this exact section
        if section_key in seen_sections:
            continue
            
        # Skip if we have too many from this document already
        if doc_counts.get(doc, 0) >= max_per_doc:
            continue
            
        # Skip if section title is empty or invalid
        if not section_title or len(section_title.strip()) < 3:
            continue
        
        seen_sections.add(section_key)
        doc_counts[doc] = doc_counts.get(doc, 0) + 1
        final_sections.append((score, section))
        
        # Stop when we have enough sections
        if len(final_sections) >= max_sections:
            break
    
    # If we don't have enough sections, relax the per-document limit
    if len(final_sections) < max_sections and len(ranked_sections) > len(final_sections):
        remaining_slots = max_sections - len(final_sections)
        
        for score, section in ranked_sections:
            if remaining_slots <= 0:
                break
                
            doc = section["document"]
            section_title = section.get("section_title", "").strip()
            page = section.get("page", 1)
            section_key = (doc, section_title.lower(), page)
            
            # Skip already included sections
            if section_key in seen_sections:
                continue
                
            if section_title and len(section_title.strip()) >= 3:
                seen_sections.add(section_key)
                final_sections.append((score, section))
                remaining_slots -= 1

    # Generate extracted_sections
    for rank, (score, section) in enumerate(final_sections, start=1):
        output["extracted_sections"].append({
            "document": section["document"],
            "section_title": section.get("section_title", "Unknown Section"),
            "importance_rank": rank,
            "page_number": section.get("page", 1)
        })

    # Generate subsection_analysis
    for score, section in final_sections:
        # Use summary if available, otherwise use context, otherwise use section title
        refined_text = (
            section.get("summary", "") or 
            section.get("context", "") or 
            section.get("section_title", "")
        ).strip()
        
        # Ensure refined text is meaningful
        if not refined_text or len(refined_text.split()) < 5:
            refined_text = f"Section covering: {section.get('section_title', 'Unknown topic')}"
        
        # Limit length of refined text
        words = refined_text.split()
        if len(words) > 100:
            refined_text = " ".join(words[:100]) + "..."
        
        output["subsection_analysis"].append({
            "document": section["document"],
            "refined_text": refined_text,
            "page_number": section.get("page", 1)
        })

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Write output file
    output_file = os.path.join(output_path, "combined_output.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Output written to: {output_file}")
    print(f"[INFO] Included {len(output['extracted_sections'])} sections from {len(set(s['document'] for s in output['extracted_sections']))} documents")