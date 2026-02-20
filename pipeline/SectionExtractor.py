import os
import re
import json
import math
import tempfile
import torch
import fitz # PyMuPDF
from PIL import Image
from PostProcess import PostProcess
from paddleocr import FormulaRecognition
from transformers import AutoModel, AutoTokenizer

class SectionCrop:
    @staticmethod
    def crop(coordinates):
        if not coordinates:
            return []

        input_path = coordinates[0]["input_path"]
        doc = fitz.open(input_path)

        dpi = 250
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)

        cropped_images = []
        for page_data in coordinates:
            page_idx = page_data["page_idx"]
            if page_idx >= len(doc):
                continue

            page = doc[page_idx]

            for box in page_data["boxes"]:
                x0, y0, x1, y1 = box["pdf_bbox"]
                clip = fitz.Rect(x0, y0, x1, y1)

                pix = page.get_pixmap(matrix=mat, clip=clip)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                cropped_images.append({
                    "page_idx": page_idx,
                    "order": box["order"],
                    "label": box["label"],
                    "image": img
                })

        doc.close()
        return cropped_images

    @staticmethod
    def save_images(cropped_images, output_path, pdf_name):
        os.makedirs(output_path, exist_ok=True)
        for i, crop_data in enumerate(cropped_images, start=1):
            label = crop_data["label"]
            image = crop_data["image"]
            
            filename = f"{pdf_name}_{label}_{i}.png"
            image_path = os.path.join(output_path, filename)
            image.save(image_path)


class MathExtract:
    def __init__(self, model="PP-FormulaNet_plus-L"):
        self.model = FormulaRecognition(model_name=model)
        self.results = None

    def extract(self, math_coordinates):
        cropped = SectionCrop.crop(math_coordinates)
        images = [c["image"] for c in cropped]
        output = self.model.predict(input=images, batch_size=1)
        self.results = output
        return output

    def save_results(self, output_path):
        pass

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
        
        for page_data in layout_results:
            page_idx = page_data.get("page_idx", 0)
            if page_idx >= len(doc): break
            
            page = doc[page_idx]
            boxes = page_data.get("boxes", [])
            
            box_contents = self.map_words_to_boxes(page, boxes)
            
            page_boxes = []
            page_empty_boxes = []
            for idx, box in enumerate(boxes):
                text_list = box_contents[idx]
                region = box.copy()
                
                if text_list:
                    raw_text = " ".join(text_list)
                    region["text"] = self.post_processor.process(raw_text)
                else:
                    region["text"] = ""
                    page_empty_boxes.append(region.copy())
                
                page_boxes.append(region)
            
            final_output.append({
                "input_path": self.input_path,
                "page_idx": page_idx,
                "image_size": page_data.get("image_size"),
                "pdf_size": page_data.get("pdf_size"),
                "boxes": page_boxes
            })
            
            if page_empty_boxes:
                empty_output.append({
                    "input_path": self.input_path,
                    "page_idx": page_idx,
                    "image_size": page_data.get("image_size"),
                    "pdf_size": page_data.get("pdf_size"),
                    "boxes": page_empty_boxes
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
            
        empty_path = os.path.join(output_path, f"{pdf_name}_text_empty_coordinates.json")
        with open(empty_path, "w", encoding="utf-8") as f:
            json.dump(self.empty_regions, f, ensure_ascii=False, indent=4)

class TableExtract:
    def __init__(self):
        self.results = None
        self.input_path = None

    def extract(self, table_coordinates):
        if not table_coordinates:
            return []
        self.input_path = table_coordinates[0]["input_path"]
        cropped = SectionCrop.crop(table_coordinates)
        # TODO: feed cropped images to TableFormer model
        self.results = cropped
        return cropped

    def save_results(self, output_path):
        if not self.results:
            return
        pdf_name = os.path.splitext(os.path.basename(self.input_path))[0]
        SectionCrop.save_images(self.results, output_path, pdf_name)

class VLMExtract:
    def __init__(self):
        self.input_path = None
        self.cropped_images = None

        model_name = 'deepseek-ai/DeepSeek-OCR-2' 

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True)

        self.dtype = torch.float16
        if torch.cuda.is_available():
            self.device = "cuda"
            if torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
        else:
            self.device = "cpu"

        self.model = self.model.eval().to(self.device).to(self.dtype)

        self.prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        # self.image_file = 'your_image.jpg'
        # self.output_path = 'your/output/dir'        


    def partial_extract(self, empty_coordinates):
        self.cropped_images = SectionCrop.crop(empty_coordinates)
        if self.cropped_images:
            self.input_path = empty_coordinates[0]["input_path"]
        # TODO: run VLM inference on self.cropped_images

    def full_extract(self):
        pass


    def save_results(self, output_path):
        if not self.cropped_images:
            return
        
        os.makedirs(output_path, exist_ok=True)
        
        pdf_name = os.path.splitext(os.path.basename(self.input_path))[0]
        
        for crop_data in self.cropped_images:
            page_idx = crop_data["page_idx"]
            order = crop_data["order"]
            label = crop_data["label"]
            image = crop_data["image"]
            
            filename = f"{pdf_name}_p{page_idx}_o{order}_{label}.png"
            image_path = os.path.join(output_path, filename)
            image.save(image_path)

    # VLM
    def extract(self):
        pass

        res = self.model.infer(self.tokenizer, prompt=self.prompt, image_file=self.image_file, output_path = self.output_path, base_size = 1024, image_size = 768, crop_mode = True, save_results = True, test_compress = True)