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

    def has_alligned_text_layer(self, input_path, layout_coordinates):
        return True

    # This function will check the empty text_results_empty and if there are boxes one inside the other it will keep only the small boxes inside
    def has_overlapping_boxes(self):
        return True