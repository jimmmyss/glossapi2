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

    def detect(self, input_path, pdf):
        self.output = self.model.predict(input_path, batch_size=1, layout_nms=True)
        filtered_output = []

        for i, res in enumerate(self.output):
            page_json = res.json
            pdf_page = pdf.pages[i]
            
            # 1. Get dimensions for the current page
            img_h, img_w = res["input_img"].shape[:2]
            pdf_w, pdf_h = pdf_page.width, pdf_page.height
            
            # 2. Add resolution info to the 'res' block
            page_json["res"]["image_size"] = [img_w, img_h]
            page_json["res"]["pdf_size"] = [float(pdf_w), float(pdf_h)]
            
            # 3. Calculate ratios
            x_scale = pdf_w / img_w
            y_scale = pdf_h / img_h

            # 4. Filter and Calculate Box Math
            final_boxes = []
            for box_obj in page_json["res"]["boxes"]:
                if box_obj["label"] not in self.unwanted_labels and box_obj["score"] >= self.threshold:
                    coords = box_obj.get("coordinate")
                    pdf_bbox = [coords[0] * x_scale, coords[1] * y_scale, coords[2] * x_scale, coords[3] * y_scale]
                    new_box = {"pdf_bbox": pdf_bbox, "box": coords, "label": box_obj["label"], "score": box_obj["score"], "cls_id": box_obj.get("cls_id")}

                    final_boxes.append(new_box)

            page_json["res"]["boxes"] = final_boxes
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
