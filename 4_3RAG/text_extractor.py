import requests
from bs4 import BeautifulSoup

def scrape_webpage(url):
    """
    Scrapes a Wikipedia page and extracts text content.
    """
    try:
        # Make the request
        response = requests.get(url)
        
        if response.status_code == 200:
            print(f"Successfully fetched content from {url}")
            
            # Parse the HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the main content div
            content_div = soup.find('div', class_='mw-parser-output')
            
            if content_div:
                # Extract all paragraph tags
                paragraphs = content_div.find_all('p')
                
                # Join all paragraph text with blank lines
                article_text = '\n\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                
                # Write to file
                with open('Selected_Document.txt', 'w', encoding='utf-8') as f:
                    f.write(article_text)
                
                print("Successfully saved content to Selected_Document.txt")
                return article_text
            else:
                print("Could not find main content div")
                return None
                
        else:
            print(f"Failed to fetch content. Status code: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def extract_text_from_pdf(pdf_path):
    """
    Alternative function to extract text from PDF files.
    """
    try:
        import PyPDF2
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from all pages
            full_text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                full_text += page.extract_text() + "\n\n"
            
            # Clean up extra whitespace
            import re
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            
            # Write to file
            with open('Selected_Document.txt', 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            print("Successfully extracted PDF content to Selected_Document.txt")
            return full_text
            
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def main():
    # Wikipedia URL - Artificial Intelligence article
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    
    print("Starting document extraction...")
    print(f"Source: {url}")
    
    # Scrape the webpage
    content = scrape_webpage(url)
    
    if content:
        print(f"Extraction complete. Document length: {len(content)} characters")
    else:
        print("Extraction failed.")

if __name__ == '__main__':
    main()
