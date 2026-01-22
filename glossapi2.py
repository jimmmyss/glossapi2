import json
import os
import copy
from paddleocr import LayoutDetection

model = LayoutDetection(model_name="PP-DocLayoutV2")
output = model.predict("test2.pdf", batch_size=1, layout_nms=True)

# Save the results
# List holding all of the individual results
all_results = []
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


# ------- Later these wont be needed as these will stay in ram -------

# Check if the folder exists, if not, create it
output_dir = "./output/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Now this will work without an error
with open(os.path.join(output_dir, "all_results.json"), "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)
with open(os.path.join(output_dir, "filtered_results.json"), "w", encoding="utf-8") as f:
    json.dump(filtered_results, f, ensure_ascii=False, indent=4)    

