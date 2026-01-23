from LayoutDetector import LayoutDetector

def main():
    detector = LayoutDetector()
    results = detector.detect("pdfs/test2.pdf")
    detector.save_results("output")

if __name__ == "__main__":
    main()
