import fitz  # PyMuPDF
import argparse

def extract_text_from_pdf(pdf_path, pages=None):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
    # If pages not specified, process all pages
    if pages is None:
        pages = range(len(doc))
    else:
        # Convert pages string to list of integers
        # Example: "1-3,5,7-9" -> [1,2,3,5,7,8,9]
        page_numbers = []
        for part in pages.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                page_numbers.extend(range(start-1, end))
            else:
                page_numbers.append(int(part)-1)
        pages = page_numbers

    text = []
    # Extract text from specified pages
    for page_num in pages:
        if page_num < len(doc):
            page = doc[page_num]
            text.append(page.get_text())

    return 