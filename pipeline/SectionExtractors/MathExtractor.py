import fitz
from paddleocr import FormulaRecognition

class MathExtract:
    def __init__(self, model="PP-FormulaNet_plus-L"):
        self.model = FormulaRecognition(model_name=model)

    def extract(self, math_coordinates):
        output = self.model.predict(input=math_coordinates, batch_size=1)
        return output

    def save_results(self, output_path):
        pass