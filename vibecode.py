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
    
    # Get image dimensions from input_img (shape is H, W, C)
    input_img = res["input_img"]
    img_h, img_w = input_img.shape[:2]
    page_json["res"]["img_w"] = img_w
    page_json["res"]["img_h"] = img_h
    
    all_results.append(copy.deepcopy(page_json)) # Deep copy to prevent modification of the original because it is a pointer

    page_json["res"]["boxes"] = [
        box for box in page_json["res"]["boxes"] 
        if box["label"] not in unwanted_labels and box["score"] >= threshold
    ]
    filtered_results.append(page_json)

final_output = []

with pdfplumber.open(input_path) as pdf:
    for i, page_data in enumerate(filtered_results):
        pdf_page = pdf.pages[i]
        pdf_w, pdf_h = pdf_page.width, pdf_page.height

        # Get image size from PP-DocLayout
        res_meta = page_data.get("res", {})
        img_w = res_meta.get("img_w", 1024.0)
        img_h = res_meta.get("img_h", 1024.0)

        x_scale = pdf_w / img_w
        y_scale = pdf_h / img_h

        page_extractions = {"page": i + 1, "regions": []}

        # Keep original order from PP-DocLayout (has built-in reading order detection)
        boxes = page_data["res"]["boxes"]

        for box_obj in boxes:
            coords = box_obj.get("box") or box_obj.get("coordinate")
            if not coords:
                continue

            # Original image coordinates
            x0, y0, x1, y1 = coords

            # Convert to PDF coordinates (both use top-left origin, no Y-flip needed)
            bbox_pdf = (
                x0 * x_scale,
                y0 * y_scale,
                x1 * x_scale,
                y1 * y_scale
            )

            try:
                cropped = pdf_page.crop(bbox_pdf, strict=False)
                text = cropped.extract_text(x_tolerance=1.5, y_tolerance=1.5)

                if text and text.strip():
                    page_extractions["regions"].append({
                        "label": box_obj.get("label"),
                        "confidence": box_obj.get("score"),
                        "text": text.strip()
                    })
            except Exception as e:
                print(f"Warning: Could not extract text for box {coords} on page {i+1}: {e}")
                continue

        final_output.append(page_extractions)


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

