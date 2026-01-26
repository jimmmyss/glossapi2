from Analyzer import Analyze
from LayoutDetector import LayoutDetect
from SectionExtractors.TextExtractor import TextExtract
from SectionExtractors.TableExtractor import TableExtract
from SectionExtractors.MathExtractor import MathExtract
from SectionExtractors.VLMExtractor import VLMExtract

def main():
    input_path = "pdfs/test2.pdf"

    analyzer = Analyze()

    if analyzer.has_text_layer(input_path):
        detector = LayoutDetect()
        layout_coordinates = detector.detect(input_path)
        text_coordinates, table_coordinates, math_coordinates = detector.filter(layout_coordinates)
        detector.save_results("output") # For visual debugging

        text_extractor = TextExtract()
        text_extractor.extract(input_path, text_coordinates)
        text_extractor.save_results("output") # For visual debugging

        

        # if analyzer.has_alligned_text_layer(input_path):
        #     extractor = TextExtract()
        #     extractor.extract(input_path, results)
        #     extractor.save_results("output") # For visual debugging
        # else:
        #     print("Bad text layer")

    else:
        print("No text found in the PDF.") 
        # VLM

if __name__ == "__main__":
    main()