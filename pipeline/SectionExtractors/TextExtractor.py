import os
import json
import pdfplumber

class TextExtract:
    def __init__(self):
        self.extracted_text = None

    def extract(self, input_path, results):
        final_output = []

        with pdfplumber.open(input_path) as pdf:
            for i, page_data in enumerate(results):
                pdf_page = pdf.pages[i]

                page_extractions = {"page": i + 1, "regions": []}

                # Get boxes from layout detection results
                boxes = page_data["res"]["boxes"]

                for box_obj in boxes:
                    # Use pre-calculated PDF coordinates from LayoutDetector
                    pdf_bbox = box_obj.get("pdf_bbox")
                    if not pdf_bbox:
                        continue

                    try:
                        cropped = pdf_page.crop(pdf_bbox, strict=False)
                        text = cropped.extract_text(x_tolerance=1.5, y_tolerance=1.5)

                        if text and text.strip():
                            page_extractions["regions"].append({
                                "label": box_obj.get("label"),
                                "confidence": box_obj.get("score"),
                                "text": text.strip()
                            })
                    except Exception as e:
                        print(f"Warning: Could not extract text for box {pdf_bbox} on page {i+1}: {e}")
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