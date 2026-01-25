import os
import json
import pdfplumber
from paddleocr import LayoutDetection

class LayoutDetect:
    TEXT_LABELS = {"text", "title", "reference", "paragraph", "header", "abstract", "table_caption", "table_footnote", "formula_caption","figure_title"}
    TABLE_LABELS = {"table"}
    MATH_LABELS = {"formula", "equation", "inline_formula", "displayed_formula"}
    UNWANTED_LABELS = {"aside_text", "header_image", "footer_image", "formula_number", "number", "seal", "image", "content", "footnote", "chart"}
    THRESHOLD = 0.65
    
    def __init__(self, model = "PP-DocLayoutV2"):
        self.model = LayoutDetection(model_name=model)

    def detect(self, input_path):
        output = self.model.predict(input_path, batch_size=4, layout_nms=True)
        detection_coordinates = []
        self.input_path = input_path

        with pdfplumber.open(input_path) as pdf:
            for i, res in enumerate(output):
                page_json = res.json
                pdf_page = pdf.pages[i]
                
                raw_boxes = page_json["res"].pop("boxes", [])
                
                img_h, img_w = res["input_img"].shape[:2]

                pdf_w, pdf_h = pdf_page.width, pdf_page.height
                
                page_json["res"]["image_size"] = [img_w, img_h]
                page_json["res"]["pdf_size"] = [float(pdf_w), float(pdf_h)]
                
                x_scale, y_scale = pdf_w / img_w, pdf_h / img_h

                processed_boxes = []
                for order_idx, box_obj in enumerate(raw_boxes):
                    coords = box_obj["coordinate"]
                    pdf_bbox = [coords[0] * x_scale, coords[1] * y_scale, coords[2] * x_scale, coords[3] * y_scale]
                    processed_boxes.append({
                        "order": order_idx,
                        "pdf_bbox": pdf_bbox, 
                        "box": coords, 
                        "label": box_obj["label"], 
                        "score": box_obj["score"], 
                        "cls_id": box_obj["cls_id"]
                    })

                page_json["input_path"] = input_path
                page_json["page_idx"] = i
                page_json["res"]["boxes"] = processed_boxes
                detection_coordinates.append(page_json)   

        self.detection_coordinates = detection_coordinates
        return detection_coordinates

    def split(self, detection_coordinates=None):
        detection_coordinates = self.detection_coordinates
        
        text_coordinates = []
        table_coordinates = []
        math_coordinates = []
        
        for page_data in detection_coordinates:
            page_info = {
                "input_path": page_data.get("input_path"),
                "page_idx": page_data.get("page_idx"),
                "res": {
                    "image_size": page_data["res"].get("image_size"),
                    "pdf_size": page_data["res"].get("pdf_size"),
                    "boxes": []
                }
            }
            
            text_page = {**page_info, "res": {**page_info["res"], "boxes": []}}
            table_page = {**page_info, "res": {**page_info["res"], "boxes": []}}
            math_page = {**page_info, "res": {**page_info["res"], "boxes": []}}
            
            for box in page_data["res"].get("boxes", []):
                label = box["label"]
                score = box["score"]
                
                # Skip low confidence and unwanted labels
                if score < self.THRESHOLD or label in self.UNWANTED_LABELS:
                    continue
                
                # Categorize by label type
                if label in self.TEXT_LABELS:
                    text_page["res"]["boxes"].append(box)
                elif label in self.TABLE_LABELS:
                    table_page["res"]["boxes"].append(box)
                elif label in self.MATH_LABELS:
                    math_page["res"]["boxes"].append(box)
            
            text_coordinates.append(text_page)
            table_coordinates.append(table_page)
            math_coordinates.append(math_page)
        
        self.text_coordinates = text_coordinates
        self.table_coordinates = table_coordinates
        self.math_coordinates = math_coordinates
        
        return text_coordinates, table_coordinates, math_coordinates

    # For visual debugging
    def save_results(self, output_path):
        if self.detection_coordinates is None:
            return

        os.makedirs(output_path, exist_ok=True)
        
        # Get PDF name from input_path in detection results
        input_path = self.detection_coordinates[0].get("input_path", "output")
        pdf_name = os.path.splitext(os.path.basename(input_path))[0]
            
        # Save detection results JSON
        detection_path = os.path.join(output_path, f"{pdf_name}_detection_coordinates.json")
        with open(detection_path, "w", encoding="utf-8") as f:
            json.dump(self.detection_coordinates, f, ensure_ascii=False, indent=4)
        
        # Save split results if available
        if self.text_coordinates is not None:
            text_path = os.path.join(output_path, f"{pdf_name}_text_coordinates.json")
            with open(text_path, "w", encoding="utf-8") as f:
                json.dump(self.text_coordinates, f, ensure_ascii=False, indent=4)
                
        if self.table_coordinates is not None:
            table_path = os.path.join(output_path, f"{pdf_name}_table_coordinates.json")
            with open(table_path, "w", encoding="utf-8") as f:
                json.dump(self.table_coordinates, f, ensure_ascii=False, indent=4)
                
        if self.math_coordinates is not None:
            math_path = os.path.join(output_path, f"{pdf_name}_math_coordinates.json")
            with open(math_path, "w", encoding="utf-8") as f:
                json.dump(self.math_coordinates, f, ensure_ascii=False, indent=4)