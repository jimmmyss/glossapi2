import os
import json
import pymupdf # PyMuPDF
import tempfile
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

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


    def partial_extract(self, empty_regions):

        self.input_path = empty_regions[0]["input_path"]
        doc = fitz.open(self.input_path)

        dpi = 250
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)

        cropped_images = []
        for page_data in empty_regions:
            page_idx = page_data["page"]
            if page_idx >= len(doc):
                continue

            page = doc[page_idx]

            page_crops = []
            for region in page_data["regions"]:
                # pdf_bbox is already in PDF points — matches PyMuPDF coordinates
                x0, y0, x1, y1 = region["pdf_bbox"]
                clip = fitz.Rect(x0, y0, x1, y1)

                # Render only the clipped region at target DPI
                pix = page.get_pixmap(matrix=mat, clip=clip)

                # Convert pixmap to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                page_crops.append({
                    "order": region["order"],
                    "label": region["label"],
                    "image": img
                })
 
            cropped_images.append({
                "page": page_idx,
                "crops": page_crops
            })

        doc.close()
        self.cropped_images = cropped_images
        return cropped_images

    def full_extract(self):
        pass


    def save_results(self, output_path):
        if not self.cropped_images:
            return
        
        os.makedirs(output_path, exist_ok=True)
        
        pdf_name = os.path.splitext(os.path.basename(self.input_path))[0]
        
        for page_data in self.cropped_images:
            page_idx = page_data["page"]
            for crop_data in page_data["crops"]:
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
