import os
import json
import pdf2image

class VLMExtract:
    def __init__(self):
        self.input_path = None
        self.cropped_images = None

    def coordinate_extraction(self, empty_regions):
        # Get input_path from the first page in empty_regions
        self.input_path = empty_regions[0]["input_path"]
        
        # Convert PDF pages to images
        pdf_images = pdf2image.convert_from_path(self.input_path, dpi=250)
        
        cropped_images = []
        for page_data in empty_regions:
            page_idx = page_data["page"]
            if page_idx >= len(pdf_images):
                continue
            
            page_image = pdf_images[page_idx]
            actual_width, actual_height = page_image.size
            
            # Get original detection image size from JSON
            orig_width, orig_height = page_data["image_size"]
            
            # Scale factors from detection image to pdf2image output
            scale_x = actual_width / orig_width
            scale_y = actual_height / orig_height
            
            page_crops = []
            for region in page_data["regions"]:
                # Use box coordinates (from detection) and scale to actual image
                box = region["box"]
                x0, y0, x1, y1 = box
                
                x0 = int(x0 * scale_x)
                y0 = int(y0 * scale_y)
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                
                cropped = page_image.crop((x0, y0, x1, y1))
                
                page_crops.append({
                    "order": region["order"],
                    "label": region["label"],
                    "image": cropped
                })
            
            cropped_images.append({
                "page": page_idx,
                "crops": page_crops
            })
        
        self.cropped_images = cropped_images
        return cropped_images

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
