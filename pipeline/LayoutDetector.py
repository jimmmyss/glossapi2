import os
import json
import fitz
from paddleocr import LayoutDetection

class LayoutDetect:
    TEXT_LABELS = {"text", "title", "reference", "paragraph", "header", "abstract", "table_caption", "table_footnote", "formula_caption","figure_title"}
    TABLE_LABELS = {"table"}
    MATH_LABELS = {"formula", "equation", "inline_formula", "displayed_formula"}
    UNWANTED_LABELS = {"aside_text", "header_image", "footer_image", "formula_number", "number", "seal", "image", "content", "footnote", "chart"}
    
    def __init__(self, model = "PP-DocLayoutV3"):
        self.model = LayoutDetection(model_name=model)

    def detect(self, input_path):
        output = self.model.predict(input_path, batch_size=4, layout_nms=True, threshold=0.2)

        layout_coordinates = []
        self.input_path = input_path

        doc = fitz.open(input_path)

        for i, res in enumerate(output):
            page_json = res.json
            page = doc[i]
                    
            # The model's raw detections (in pixels)
            raw_boxes = page_json["res"].pop("boxes", [])
                    
            # Image dimensions (what the AI saw)
            img_h, img_w = res["input_img"].shape[:2]

            # PDF dimensions (the physical document size)
            pdf_w = page.rect.width
            pdf_h = page.rect.height
                    
            # Calculate scale factors
            x_scale, y_scale = pdf_w / img_w, pdf_h / img_h

            processed_boxes = []
            for order_idx, box_obj in enumerate(raw_boxes):
                coords = box_obj["coordinate"] # [x1, y1, x2, y2]
                        
                # Scale coordinates to PDF points
                pdf_bbox = [
                    coords[0] * x_scale, 
                    coords[1] * y_scale, 
                    coords[2] * x_scale, 
                    coords[3] * y_scale
                ]
                        
                processed_boxes.append({
                    "order": order_idx,
                    "pdf_bbox": pdf_bbox, 
                    "box": coords, 
                    "label": box_obj["label"], 
                    "score": box_obj["score"], 
                    "cls_id": box_obj["cls_id"]
                })

            layout_coordinates.append({
                "input_path": input_path,
                "page_idx": i,
                "image_size": [img_w, img_h],
                "pdf_size": [float(pdf_w), float(pdf_h)],
                "boxes": processed_boxes
            })   

        doc.close()

        self.layout_coordinates = layout_coordinates
        self.model_output = output
        return layout_coordinates

    def filter(self, layout_coordinates=None):
        layout_coordinates = self.layout_coordinates
        
        text_coordinates = []
        table_coordinates = []
        math_coordinates = []
        
        for page_data in layout_coordinates:
            page_base = {
                "input_path": page_data.get("input_path"),
                "page_idx": page_data.get("page_idx"),
                "image_size": page_data.get("image_size"),
                "pdf_size": page_data.get("pdf_size"),
            }
            
            text_page = {**page_base, "boxes": []}
            table_page = {**page_base, "boxes": []}
            math_page = {**page_base, "boxes": []}
            
            for box in page_data.get("boxes", []):
                label = box["label"]
                
                # Skip unwanted labels
                if label in self.UNWANTED_LABELS:
                    continue
                
                # Categorize by label type
                if label in self.TEXT_LABELS:
                    text_page["boxes"].append(box)
                elif label in self.TABLE_LABELS:
                    table_page["boxes"].append(box)
                elif label in self.MATH_LABELS:
                    math_page["boxes"].append(box)
            
            # Only include pages that have detections
            if text_page["boxes"]:
                text_coordinates.append(text_page)
            if table_page["boxes"]:
                table_coordinates.append(table_page)
            if math_page["boxes"]:
                math_coordinates.append(math_page)
        
        self.text_coordinates = text_coordinates
        self.table_coordinates = table_coordinates
        self.math_coordinates = math_coordinates
        
        return text_coordinates, table_coordinates, math_coordinates

    # For visual debugging
    def save_results(self, output_path):
        if self.layout_coordinates is None:
            return

        os.makedirs(output_path, exist_ok=True)
        
        # Get PDF name from input_path in detection results
        input_path = self.layout_coordinates[0].get("input_path", "output")
        pdf_name = os.path.splitext(os.path.basename(input_path))[0]
        
        # Save the photos
        for res in self.model_output:
            res.save_to_img(save_path=output_path)

        # Save detection results JSON
        detection_path = os.path.join(output_path, f"{pdf_name}_layout_coordinates.json")
        with open(detection_path, "w", encoding="utf-8") as f:
            json.dump(self.layout_coordinates, f, ensure_ascii=False, indent=4)
        
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