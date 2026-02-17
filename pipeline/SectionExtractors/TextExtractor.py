import os
import re
import json
import math
import fitz
from PostProcess import PostProcess

class TextExtract:
    def __init__(self):
        self.extracted_text = None
        self.empty_regions = None
        self.input_path = None
        self.post_processor = PostProcess()

    def map_words_to_boxes(self, page, layout_boxes):
        all_words = page.get_text("words")
        
        box_contents = {i: [] for i in range(len(layout_boxes))}

        for word in all_words:
            wx0, wt, wx1, wb, text = word[0], word[1], word[2], word[3], word[4]
            
            word_area = (wx1 - wx0) * (wb - wt) + 1e-9
            
            candidates = []
            for idx, box in enumerate(layout_boxes):
                bx0, bt, bx1, bb = box["pdf_bbox"]
                
                ix0 = max(wx0, bx0)
                it  = max(wt, bt)
                ix1 = min(wx1, bx1)
                ib  = min(wb, bb)
                
                if ix1 > ix0 and ib > it:
                    inter_area = (ix1 - ix0) * (ib - it)
                    iow_score = inter_area / word_area
                    box_area = (bx1 - bx0) * (bb - bt)
                    candidates.append({
                        "idx": idx,
                        "iow": iow_score,
                        "area": box_area
                    })
            
            if candidates:
                best_box = sorted(candidates, key=lambda x: (-x["iow"], x["area"]))[0]
                box_contents[best_box["idx"]].append(text)

        return box_contents

    def extract(self, layout_results):
        final_output = []
        empty_output = []
        if not layout_results:
            return [], []
                
        self.input_path = layout_results[0]["input_path"]
            
        doc = fitz.open(self.input_path)
            
        for i, page_data in enumerate(layout_results):
            if i >= len(doc): break
                
            page = doc[i]
            boxes = page_data.get("res", {}).get("boxes", [])
            image_size = page_data.get("res", {}).get("image_size")
            pdf_size = page_data.get("res", {}).get("pdf_size")
                
            box_contents = self.map_words_to_boxes(page, boxes)
                
            page_regions = []
            page_empty_regions = []
            for idx, box in enumerate(boxes):
                text_list = box_contents[idx]
                region = box.copy()
                    
                if text_list:
                    raw_text = " ".join(text_list)
                    region["text"] = self.post_processor.process(raw_text)
                else:
                    region["text"] = ""
                    page_empty_regions.append(region.copy())
                    
                page_regions.append(region)
                
                final_output.append({
                    "input_path": self.input_path,
                    "page": i,
                    "image_size": image_size,
                    "pdf_size": pdf_size,
                    "regions": page_regions
                })
                
                if page_empty_regions:
                    empty_output.append({
                        "input_path": self.input_path,
                        "page": i,
                        "image_size": image_size,
                        "pdf_size": pdf_size,
                        "regions": page_empty_regions
                    })

        doc.close()
        self.extracted_text = final_output
        self.empty_regions = empty_output
        return final_output, empty_output

    def save_results(self, output_path):
        if not self.extracted_text:
            return

        os.makedirs(output_path, exist_ok=True)
            
        pdf_name = os.path.splitext(os.path.basename(self.input_path))[0]
            
        json_path = os.path.join(output_path, f"{pdf_name}_text_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.extracted_text, f, ensure_ascii=False, indent=4)
            
        empty_path = os.path.join(output_path, f"{pdf_name}_text_results_empty.json")
        with open(empty_path, "w", encoding="utf-8") as f:
            json.dump(self.empty_regions, f, ensure_ascii=False, indent=4)
