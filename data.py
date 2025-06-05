import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from PIL import Image
from transformers import pipeline
import re
import os
import logging
import urllib.parse
from lxml import etree
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize image captioning model (BLIP)
try:
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", use_fast=True)
except Exception as e:
    logger.warning(f"Failed to load BLIP model: {e}. Using placeholder captions.")
    captioner = None

# Global counter for unique chunk IDs
global_chunk_counter = 0

# Validate LaTeX
def validate_latex(text):
    try:
        if not text or text.strip() in ['=', '', '{}']:
            return False
        stripped = text.strip('$').strip()
        if not re.search(r'\\[a-zA-Z]+|[+\-*/=∑∫∆]', stripped):
            return False
        brace_count = 0
        for char in stripped:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
            if brace_count < 0:
                return False
        return brace_count == 0
    except Exception as e:
        logger.error(f"LaTeX validation error: {e}, Input: {text}")
        return False

# Clean LaTeX
def clean_latex(text):
    try:
        if not text or text.strip() in ['=', '', '$$']:
            return None
        text = text.replace('\\\\', '\\').strip()
        text = re.sub(r'\${2,}', '$', text)
        text = text.strip('$').strip()
        replacements = [
            (r'\s*~\s*([a-zA-Z])', r'\\tilde{\1}'),
            (r"C'\s*_t", r'\\tilde{C}_t'),
            (r'\[([^\]]+),\s*([^\]]+)\]', r'[\1, \2]'),
            (r'\\begin{cases}(.*?)\\end{cases}', r'\\begin{cases}\1\\end{cases}', re.DOTALL),
            (r'\\begin{align\*}(.*?)\\end{align\*}', r'\\begin{align*}\1\\end{align*}', re.DOTALL)
        ]
        for pattern, repl in replacements:
            text = re.sub(pattern, repl, text)
        if not validate_latex(text):
            return None
        return f"$${text}$$"
    except Exception as e:
        logger.error(f"Error cleaning LaTeX: {e}, Input: {text}")
        return None

# Generate unique source ID
def generate_source_id(url):
    return hashlib.md5(url.encode()).hexdigest()

# Step 1: Data Collection
def scrape_html(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        parser = etree.HTMLParser()
        tree = etree.fromstring(response.content, parser)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove boilerplate
        for elem in soup.select('nav, footer, .sidebar, .comments'):
            elem.decompose()

        # Extract text
        content_area = soup.select_one('article, .post-content, .entry-content, main')
        text = content_area.get_text(separator=' ', strip=True) if content_area else soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text).strip()

        # Extract LaTeX equations
        equations = []
        for script in soup.find_all('script', type='math/tex'):
            eq = script.text.strip()
            if validate_latex(eq):
                equations.append(clean_latex(eq) or eq)
        for span in soup.find_all('span', class_=['math', 'mjx-chtml']):
            eq = span.text.strip()
            if validate_latex(eq):
                equations.append(clean_latex(eq) or eq)
        for math in tree.xpath('//math'):
            try:
                latex = etree.tostring(math, encoding='unicode')
                latex = re.sub(r'<[^>]+>', '', latex).strip()
                if validate_latex(latex):
                    equations.append(clean_latex(latex) or latex)
            except Exception as e:
                logger.warning(f"Failed to process <math> tag: {e}")

        # Extract and save images
        images = []
        os.makedirs("./images/html", exist_ok=True)
        for img in soup.find_all('img'):
            img_url = img.get('src')
            if img_url:
                img_url = urllib.parse.urljoin(url, img_url)
                try:
                    img_response = requests.get(img_url, timeout=5)
                    img_response.raise_for_status()
                    global global_chunk_counter
                    img_name = f"{global_chunk_counter}_{os.path.basename(img_url).split('?')[0]}"
                    img_name = re.sub(r'[^\w\.-]', '_', img_name)
                    img_path = f"./images/html/{img_name}"
                    with open(img_path, 'wb') as f:
                        f.write(img_response.content)
                    parent = img.find_parent(['p', 'div'])
                    context_text = parent.get_text(strip=True)[:200] if parent else ""
                    images.append({
                        'url': img_url,
                        'path': img_path,
                        'context_text': context_text
                    })
                    global_chunk_counter += 1
                except Exception as e:
                    logger.warning(f"Failed to download image {img_url}: {e}")

        # Advanced section identification
        sections = []
        current_section = "Introduction"
        section_index = 0
        paragraph_index = 0
        headings = soup.find_all(['h1', 'h2', 'h3'])
        content_elements = soup.find_all(['p', 'div'])
        for tag in headings + content_elements:
            if tag.name in ['h1', 'h2', 'h3']:
                current_section = tag.get_text(strip=True)
                section_index += 1
                paragraph_index = 0
            elif tag.name in ['p', 'div'] and tag.get_text(strip=True):
                paragraph_index += 1
                sections.append({
                    'section': current_section,
                    'content': tag.get_text(strip=True),
                    'content_type': 'text',
                    'section_index': section_index,
                    'paragraph_index': paragraph_index,
                    'source_id': generate_source_id(url)
                })

        logger.info(f"Scraped {len(text)} chars, {len(equations)} equations, {len(images)} images from {url}")
        return text, equations, images, sections
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return "", [], [], []

def download_and_extract_pdf(url, output_path="lstm_notes.pdf"):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(response.content)

        doc = fitz.open(output_path)
        text = ""
        images = []
        equations = []
        os.makedirs("./images/pdf", exist_ok=True)

        for page_num, page in enumerate(doc):
            page_text = page.get_text("text")
            text += page_text + "\n"

            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                if base_image:
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    global global_chunk_counter
                    img_path = f"./images/pdf/page_{page_num}_img_{global_chunk_counter}.{image_ext}"
                    with open(img_path, 'wb') as f:
                        f.write(image_bytes)
                    context_text = page.get_text("text", clip=page.rect)[:200]
                    images.append({
                        'url': f"local://{img_path}",
                        'path': img_path,
                        'context_text': context_text
                    })
                    logger.debug(f"Extracted PDF image: {img_path}")
                    global_chunk_counter += 1

            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            if pix:
                img_path = f"./images/pdf/page_{page_num}_render_{global_chunk_counter}.png"
                pix.save(img_path)
                context_text = page.get_text("text", clip=page.rect)[:200]
                images.append({
                    'url': f"local://{img_path}",
                    'path': img_path,
                    'context_text': context_text
                })
                logger.debug(f"Rendered PDF page as image: {img_path}")
                global_chunk_counter += 1

            page_blocks = page.get_text("dict")["blocks"]
            for block in page_blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            span_text = span["text"].strip()
                            if re.search(r'\\[a-zA-Z]+|[∑∫∆]', span_text) and validate_latex(span_text):
                                cleaned = clean_latex(span_text)
                                if cleaned:
                                    equations.append(cleaned)

            sections = []
            current_section = "Introduction"
            section_index = 0
            paragraph_index = 0
            lines = page_text.split('\n')
            for line in lines:
                line = line.strip()
                if re.match(r'^\d+\.\d+\.?|^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', line):
                    current_section = line
                    section_index += 1
                    paragraph_index = 0
                elif line:
                    paragraph_index += 1
                    sections.append({
                        'section': current_section,
                        'content': line,
                        'content_type': 'text',
                        'section_index': section_index,
                        'paragraph_index': paragraph_index,
                        'page_number': page_num + 1,
                        'source_id': generate_source_id(url)
                    })

        logger.info(f"Extracted {len(text)} chars, {len(equations)} equations, {len(images)} images from {output_path}")
        return text, equations, images, sections
    except Exception as e:
        logger.error(f"Error processing PDF {url}: {e}")
        return "", [], [], []

# Step 2: Preprocessing
def convert_latex_to_text(latex):
    try:
        latex_clean = latex.strip('$')
        replacements = {
            r'\\sigma': 'sigmoid',
            r'\\cdot': 'dot product',
            r'_{t-1}': ' at time t-1',
            r'\\frac{([^}]+)}{([^}]+)}': 'fraction \\1 over \\2',
            r'\\sum': 'summation',
            r'\\in': 'in',
            r'\\mathbb{R}': 'real numbers',
            r'f_t': 'forget gate',
            r'i_t': 'input gate',
            r'o_t': 'output gate',
            r'c_t': 'cell state',
            r'h_t': 'hidden state'
        }
        for pattern, repl in replacements.items():
            latex_clean = re.sub(pattern, repl, latex_clean)
        latex_clean = re.sub(r'[\{\}\\]', '', latex_clean)
        return f"Equation: {latex_clean}"
    except Exception as e:
        logger.error(f"Error converting LaTeX: {e}")
        return f"Equation: {latex}"

def generate_image_caption(img_path, context_text):
    try:
        if captioner and os.path.exists(img_path):
            img = Image.open(img_path)
            caption = captioner(img)[0]['generated_text']
            return f"{caption} (Context: {context_text})"
        return f"Diagram related to LSTMs (Context: {context_text})"
    except Exception as e:
        logger.warning(f"Error generating caption for {img_path}: {e}")
        return f"Diagram related to LSTMs (Context: {context_text})"

def preprocess_content(text, equations, images, sections, source_name, source_id):
    global global_chunk_counter
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = []

        # Process text sections
        text_chunk_ids = []
        for section in sections:
            section_chunks = text_splitter.split_text(section['content'])
            for chunk in section_chunks:
                chunk_id = f"{source_id}_text_{global_chunk_counter}"
                global_chunk_counter += 1
                text_chunk_ids.append(chunk_id)
                metadata = {
                    'source': source_name,
                    'source_id': source_id,
                    'section': section['section'],
                    'content_type': 'text',
                    'chunk_id': chunk_id,
                    'section_index': section.get('section_index', 0),
                    'paragraph_index': section.get('paragraph_index', 0)
                }
                if 'page_number' in section:
                    metadata['page_number'] = section['page_number']
                chunks.append({
                    'content': chunk,
                    'metadata': metadata
                })

        # Process equations
        for i, eq in enumerate(equations):
            eq_description = convert_latex_to_text(eq)
            chunk_id = f"{source_id}_equation_{global_chunk_counter}"
            global_chunk_counter += 1
            nearest_chunk_id = text_chunk_ids[i % len(text_chunk_ids)] if text_chunk_ids else ""
            chunks.append({
                'content': eq_description,
                'metadata': {
                    'source': source_name,
                    'source_id': source_id,
                    'section': 'Equations',
                    'content_type': 'equation',
                    'raw_content': eq,
                    'chunk_id': chunk_id,
                    'nearest_text_chunk_id': nearest_chunk_id
                }
            })

        # Process images
        for i, img in enumerate(images):
            img_description = generate_image_caption(img['path'], img['context_text'])
            chunk_id = f"{source_id}_image_{global_chunk_counter}"
            global_chunk_counter += 1
            nearest_chunk_id = text_chunk_ids[i % len(text_chunk_ids)] if text_chunk_ids else ""
            tags = ['lstm', 'diagram']
            if 'gate' in img_description.lower():
                tags.append('gate')
            if 'cell' in img_description.lower():
                tags.append('cell')
            if 'state' in img_description.lower():
                tags.append('state')
            tags_str = ','.join(tags)
            chunks.append({
                'content': img_description,
                'metadata': {
                    'source': source_name,
                    'source_id': source_id,
                    'section': 'Images',
                    'content_type': 'image',
                    'raw_content': img['path'],
                    'chunk_id': chunk_id,
                    'nearest_text_chunk_id': nearest_chunk_id,
                    'tags': tags_str
                }
            })

        logger.info(f"Preprocessed {len(chunks)} chunks for {source_name}")
        return chunks
    except Exception as e:
        logger.error(f"Error preprocessing content for {source_name}: {e}")
        return []

# Step 3: Embedding and Storing in Vector Database
def store_in_vector_db(chunks, persist_directory="./chroma_db"):
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma(
            collection_name="lstm_rag",
            embedding_function=embedding_model,
            persist_directory=persist_directory
        )

        texts = [chunk['content'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        ids = [chunk['metadata']['chunk_id'] for chunk in chunks]

        if len(ids) != len(set(ids)):
            duplicates = {id for id in ids if ids.count(id) > 1}
            logger.error(f"Found {len(duplicates)} duplicate IDs: {duplicates}")
            raise ValueError(f"Duplicate IDs found: {duplicates}")

        vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        logger.info(f"Stored {len(texts)} chunks in Chroma vector database at {persist_directory}")
        return vectorstore
    except Exception as e:
        logger.error(f"Error storing in vector database: {e}")
        return None

# Main Execution
def main():
    html_url = "https://colah.github.io/posts/2015-08-Understanding-LSTMs/"
    pdf_url = "https://deeplearning.cs.cmu.edu/S23/document/readings/LSTM.pdf"

    os.makedirs("./chroma_db", exist_ok=True)

    logger.info("Scraping HTML content...")
    html_text, html_equations, html_images, html_sections = scrape_html(html_url)
    html_source_id = generate_source_id(html_url)

    logger.info("Downloading and extracting PDF content...")
    pdf_text, pdf_equations, pdf_images, pdf_sections = download_and_extract_pdf(pdf_url)
    pdf_source_id = generate_source_id(pdf_url)

    logger.info("Preprocessing content...")
    html_chunks = preprocess_content(html_text, html_equations, html_images, html_sections, "Understanding LSTMs", html_source_id)
    pdf_chunks = preprocess_content(pdf_text, pdf_equations, pdf_images, pdf_sections, "CMU LSTM Notes", pdf_source_id)

    all_chunks = html_chunks + pdf_chunks

    logger.info("Storing in vector database...")
    vectorstore = store_in_vector_db(all_chunks)

    if vectorstore:
        logger.info("Data collection and storage completed successfully.")
    else:
        logger.error("Failed to store data in vector database.")

if __name__ == "__main__":
    main()