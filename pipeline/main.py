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
        results = detector.detect(input_path)
        detector.save_results("output") # For visual debugging

        if analyzer.has_alligned_text_layer(input_path):
            extractor = TextExtract()
            extractor.extract(input_path, results)
            extractor.save_results("output") # For visual debugging
        else:
            print("Bad text layer")

    else:
        print("No text found in the PDF.") 
        # VLM

if __name__ == "__main__":
    main()