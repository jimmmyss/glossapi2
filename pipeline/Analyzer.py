import pdfplumber

class Analyze:
    def __init__(self):
        pass

    def has_text_layer(self, input_path):
        with pdfplumber.open(input_path) as pdf:
            for page in pdf.pages:
                if len(page.chars) > 0:
                    return True
            return False

    def has_alligned_text_layer(self, input_path):
        return True
    