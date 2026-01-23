# Can be deleted later with save_to_img def 
import os
from paddleocr import LayoutDetection

class LayoutDetector:
    def __init__(self, model = "PP-DocLayoutV2", threshold = 0.65, unwanted_labels = None):
        self.model = LayoutDetection(model_name=model)
        self.threshold = threshold
        if unwanted_labels is None:
            self.unwanted_labels = {"aside_text", "header_image", "footer_image", "formula_number", "number", "seal", "image", "content", "footnote", "chart"}
        else:
            self.unwanted_labels = unwanted_labels

        # Safety for images, can be deleted later with save_to_img def
        self.output = None

    def detect(self, input_path):
        self.output = self.model.predict(input_path, batch_size=1, layout_nms=True)
        filtered_output = []

        for res in self.output:
            page_json = res.json
            
            # Extract per-page dimensions for text layer coordination
            img_h, img_w = res["input_img"].shape[:2]
            page_json["res"]["img_w"] = img_w
            page_json["res"]["img_h"] = img_h
            
            # Filter out unwanted labels and low confidence boxes
            page_json["res"]["boxes"] = [
                box for box in page_json["res"]["boxes"] 
                if box["label"] not in self.unwanted_labels and box["score"] >= self.threshold
            ]

            filtered_output.append(page_json)   

        return filtered_output

    # For visual debugging, can delete later
    def save_to_img(self, output_path):
        if self.output is None:
            return

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        for res in self.output:
            res.save_to_img(save_path=output_path)
    
