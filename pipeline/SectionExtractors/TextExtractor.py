import os
import re
import json
import pdfplumber

class TextExtract:
    def __init__(self):
        self.extracted_text = None

    def extract(self, input_path, results):
        final_output = []

        # Only text, paragraph etc will go in fix later
        with pdfplumber.open(input_path) as pdf:
            for i, page_data in enumerate(results):
                pdf_page = pdf.pages[i]

                page_extractions = {"page": i, "regions": []}

                boxes = page_data["res"]["boxes"]

                for box_obj in boxes:
                    pdf_bbox = box_obj.get("pdf_bbox")
                    if not pdf_bbox or pdf_bbox[0] >= pdf_bbox[2] or pdf_bbox[1] >= pdf_bbox[3]:
                        continue

                    try:
                        text = pdf_page.crop(pdf_bbox, strict=False).extract_text(x_tolerance_ratio=0.09) # Sweet spot for typography standards for word spacing

                        if text and text.strip():
                            #text = re.sub(r'-\s+([a-z])', r'\1', text) # Remove hyphenated words only if followed by whitespace AND lowercase letter(exam- ple -> example)
                            #text = " ".join(text.split()) # Remove whitespaces and newlines
                            page_extractions["regions"].append({
                                "label": box_obj.get("label"),
                                "confidence": box_obj.get("score"),
                                "text": text
                            })

                    except Exception as e:
                        continue

                final_output.append(page_extractions)

        self.extracted_text = final_output
        return final_output
    
    def save_results(self, output_path):
        if self.extracted_text is None:
            return
        
        os.makedirs(output_path, exist_ok=True)
        
        json_path = os.path.join(output_path, "extracted_text.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.extracted_text, f, ensure_ascii=False, indent=4) 
