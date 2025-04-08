import os
from PyPDF2 import PdfReader
from setfit import SetFitModel
import warnings
warnings.filterwarnings('ignore')


# Load the trained model (update the path if needed)
label2id = {"Contract": 0, "Email": 1, "Invoice": 2, "Resume": 3}
id2label = {v: k for k, v in label2id.items()}

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file that contains embedded text (non-scanned PDFs).
    """
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

def classify_pdf(pdf_path, model):
    """
    Extract text from a PDF and classify the document using the loaded model.
    """
    # Extract text directly from the PDF
    extracted_text = extract_text_from_pdf(pdf_path)

    if not extracted_text.strip():
        print("No text found in the document.")
        return None

    # The model expects a list of strings, one for each document
    prediction = model([extracted_text])
    return prediction

if __name__ == "__main__":
    model = SetFitModel.from_pretrained("./docs_classifier_model")
    # Get all PDF files in the directory
    test_dir = ".\\img_classification_testing_2-2"
    pdf_files = [f for f in os.listdir(test_dir) if f.endswith('.pdf')]

    # Process each PDF file
    for pdf_file in pdf_files:
        pdf_path = os.path.join(test_dir, pdf_file)
        print(f"\nProcessing: {pdf_file}")

        if os.path.exists(pdf_path):
            prediction = classify_pdf(pdf_path, model)
            if prediction is not None:

                resultlabel = id2label[int(prediction[0])]
                print("Document Prediction:", resultlabel)
        else:
            print(f"File not found: {pdf_path}")