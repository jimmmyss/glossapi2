# venv
python -m venv glossapi_env
source glossapi_env/bin/activate

# PP-DocLayoutV2 requirements
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install -U "paddleocr[doc-parser]"
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl

# Label_list
- abstract
- algorithm
- aside_text
- chart
- content
- display_formula
- doc_title
- figure_title
- footer
- footer_image
- footnote
- formula_number
- header
- header_image
- image
- inline_formula
- number
- paragraph_title
- reference
- reference_content
- seal
- table
- text
- vertical_text
- vision_footnote

https://huggingface.co/PaddlePaddle/PaddleOCR-VL/blob/a3dbeaddf7ff5718914d68633b269f228c286479/PP-DocLayoutV2/inference.yml
