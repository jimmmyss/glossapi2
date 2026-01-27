from Analyzer import Analyze
from LayoutDetector import LayoutDetect
from SectionExtractors.TextExtractor import TextExtract
from SectionExtractors.TableExtractor import TableExtract
from SectionExtractors.MathExtractor import MathExtract
from SectionExtractors.VLMExtractor import VLMExtract

def main():
    input_path = "pdfs/9ΩΙΝ46ΜΔΨΟ-154.pdf"

    analyzer = Analyze()

    if analyzer.has_text_layer(input_path):
        detector = LayoutDetect()
        layout_coordinates = detector.detect(input_path)
        text_coordinates, table_coordinates, math_coordinates = detector.filter(layout_coordinates)
        detector.save_results("output") # For visual debugging

        # <section>_coordinates are never truly empty so create functions that check them
        if text_coordinates is not None:
            text_extractor = TextExtract()
            text_results, text_results_empty = text_extractor.extract(text_coordinates)
            text_extractor.save_results("output") # For visual debugging
            # if text is empty then
            vlm_extractor = VLMExtract()
            vlm_extractor.coordinate_extraction(text_results_empty)
            vlm_extractor.save_results("output") # For visual debugging
    
        # if table_coordinates is not None:
        #     table_extractor = TableExtract()
        #     table_extractor.extract(input_path, table_coordinates)
        #     table_extractor.save_results("output") # For visual debugging

        # if math_coordinates is not None:
        #     math_extractor = MathExtract()
        #     math_extractor.extract(input_path, math_coordinates)
        #     math_extractor.save_results("output") # For visual debugging

    else:
        print("No text found in the PDF.") 
        # VLM

if __name__ == "__main__":
    main()