import json
import os
import copy
import pdfplumber
from paddleocr import LayoutDetection

input_path = "test2.pdf"
output_dir = "./output/"

model = LayoutDetection(model_name="PP-DocLayoutV2")
output = model.predict(input_path, batch_size=1, layout_nms=True)

# List holding all of the individual results
all_results = [] # Can delete later
# List holding all of the filtered results
filtered_results = []

threshold = 0.65
unwanted_labels = {"aside_text", "header_image", "footer_image", "formula_number", "number", "seal", "image", "content", "footnote", "chart"}

for res in output:
    # res.print() # For CLI printing
    res.save_to_img(save_path="./output/") # For saving images

    page_json = res.json
    
    all_results.append(copy.deepcopy(page_json)) # Deep copy to prevent modification of the original because it is a pointer

    page_json["res"]["boxes"] = [
        box for box in page_json["res"]["boxes"] 
        if box["label"] not in unwanted_labels and box["score"] >= threshold
    ]
    filtered_results.append(page_json)


final_output = []

# Vibe coded shit for pdfplumber change later

with pdfplumber.open(input_path) as pdf:
    for i, page_data in enumerate(filtered_results):
        pdf_page = pdf.pages[i]
        
        # 1. Get exact PDF points
        p_w, p_h = float(pdf_page.width), float(pdf_page.height)

        # 2. Get exact Paddle detection resolution from metadata
        res_meta = page_data.get("res", {})
        img_w = res_meta.get("img_w")
        img_h = res_meta.get("img_h")

        # Fallback to standard model resolution if metadata is missing
        if not img_w or not img_h:
            img_w, img_h = 1024.0, 1024.0 

        # 3. Calculate scales
        x_scale = p_w / img_w
        y_scale = p_h / img_h

        page_extractions = {"page": i + 1, "regions": []}

        # Sort by top-coordinate to ensure logical reading order in JSON
        boxes = page_data["res"]["boxes"]
        boxes.sort(key=lambda x: (x.get("box") or x.get("coordinate"))[1])

        for box_obj in boxes:
            coords = box_obj.get("box") or box_obj.get("coordinate")
            if not coords: continue

            # Apply scaling precisely
            bbox = (
                coords[0] * x_scale,
                coords[1] * y_scale,
                coords[2] * x_scale,
                coords[3] * y_scale
            )

            try:
                # strict=False allows the box to be slightly off-page without crashing
                cropped = pdf_page.crop(bbox, strict=False)
                
                # x_tolerance=1.5 fixes "PanagiotisVagenas" -> "Panagiotis Vagenas"
                text = cropped.extract_text(x_tolerance=1.5, y_tolerance=1.5)
                
                if text and text.strip():
                    page_extractions["regions"].append({
                        "label": box_obj.get("label"),
                        "confidence": box_obj.get("score"),
                        "text": text.strip()
                    })
            except Exception:
                continue
            
        final_output.append(page_extractions)

# Vibe coding ends, after the getting the results there are problems with coordinate mapping so it needs fixing

# ------- Later these wont be needed as these will stay in ram -------

# Check if the folder exists, if not, create it
output_dir = "./output/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# Save to output.json as requested
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=4, ensure_ascii=False)

# Now this will work without an error
with open(os.path.join(output_dir, "all_results.json"), "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)
with open(os.path.join(output_dir, "filtered_results.json"), "w", encoding="utf-8") as f:
    json.dump(filtered_results, f, ensure_ascii=False, indent=4)    

