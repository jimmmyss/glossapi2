import os
import re
import json
import math
import pdfplumber

class TextExtract:
    def __init__(self):
        self.extracted_text = None


    # Revisit later for better optimization
    # ----------------------------------------
    def extract_by_iow_fit(self, pdf_page, layout_boxes):
        all_words = pdf_page.extract_words(x_tolerance_ratio=0.12)
        box_contents = {i: [] for i in range(len(layout_boxes))}

        for word in all_words:
            wx0, wt, wx1, wb = word["x0"], word["top"], word["x1"], word["bottom"]
            # Add epsilon to prevent division by zero
            word_area = (wx1 - wx0) * (wb - wt) + 1e-9
            
            candidates = []
            for idx, box in enumerate(layout_boxes):
                bx0, bt, bx1, bb = box["pdf_bbox"]
                
                # Intersection coordinates
                ix0 = max(wx0, bx0)
                it  = max(wt, bt)
                ix1 = min(wx1, bx1)
                ib  = min(wb, bb)
                
                # Calculate intersection only if boxes actually touch
                if ix1 > ix0 and ib > it:
                    inter_area = (ix1 - ix0) * (ib - it)
                    iow_score = inter_area / word_area
                    
                    # We record EVERY intersection, no matter how small
                    box_area = (bx1 - bx0) * (bb - bt)
                    candidates.append({
                        "idx": idx,
                        "iow": iow_score,
                        "area": box_area
                    })
            
            if candidates:
                # Most coverage wins; if tied (e.g., both 100%), smallest box wins.
                best_box = sorted(candidates, key=lambda x: (-x["iow"], x["area"]))[0]
                box_contents[best_box["idx"]].append(word["text"])

        final_regions = []
        for idx, box in enumerate(layout_boxes):
            text_list = box_contents[idx]
            if text_list:
                raw_text = " ".join(text_list)
                # Standard hyphenation healing
                clean_text = re.sub(r'(\w+)-\s+', r'\1', raw_text)
                clean_text = " ".join(clean_text.split())
                
                final_regions.append({
                    "label": box.get("label"),
                    "confidence": box.get("score"),
                    "text": clean_text,
                    "bbox": box.get("pdf_bbox")
                })
        return final_regions
    # ----------------------------------------

    def extract(self, input_path, layout_results):
        final_output = []
        
        with pdfplumber.open(input_path) as pdf:
            for i, page_data in enumerate(layout_results):
                if i >= len(pdf.pages): break
                
                pdf_page = pdf.pages[i]
                boxes = page_data.get("res", {}).get("boxes", [])
                
                # Call the spatial assignment function
                page_regions = self.extract_by_iow_fit(pdf_page, boxes)
                
                final_output.append({
                    "page": i,
                    "regions": page_regions
                })

        self.extracted_text = final_output
        return final_output

    def save_results(self, output_path):
        if not self.extracted_text:
            return
        
        os.makedirs(output_path, exist_ok=True)
        json_path = os.path.join(output_path, "extracted_text.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.extracted_text, f, ensure_ascii=False, indent=4)
