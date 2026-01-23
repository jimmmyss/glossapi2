import os
import json
import pdfplumber
from paddleocr import LayoutDetection

class LayoutDetector:
    def __init__(self, model = "PP-DocLayoutV2", threshold = 0.65, unwanted_labels = None):
        self.model = LayoutDetection(model_name=model)
        self.threshold = threshold
        if unwanted_labels is None:
            self.unwanted_labels = {"aside_text", "header_image", "footer_image", "formula_number", "number", "seal", "image", "content", "footnote", "chart"}
        else:
            self.unwanted_labels = unwanted_labels

        self.filtered_results = None

    def detect(self, input_path):
        output = self.model.predict(input_path, batch_size=4, layout_nms=True)
        filtered_output = []

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
                for box_obj in raw_boxes:
                    if box_obj["label"] not in self.unwanted_labels and box_obj["score"] >= self.threshold:
                        coords = box_obj["coordinate"]
                        pdf_bbox = [coords[0] * x_scale, coords[1] * y_scale, coords[2] * x_scale, coords[3] * y_scale]
                        processed_boxes.append({"pdf_bbox": pdf_bbox, "box": coords, "label": box_obj["label"], "score": box_obj["score"], "cls_id": box_obj["cls_id"]})

                page_json["res"]["boxes"] = processed_boxes
                filtered_output.append(page_json)   

        self.filtered_results = filtered_output
        self.output = output
        return filtered_output

    # For visual debugging
    def save_results(self, output_path):
        if self.output is None:
            return

        os.makedirs(output_path, exist_ok=True)
        
        # Save visual images 
        for res in self.output:
            res.save_to_img(save_path=output_path)
            
        # Save JSON with coordinates
        json_path = os.path.join(output_path, "filtered_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.filtered_results, f, ensure_ascii=False, indent=4)
        
        self.output = None
