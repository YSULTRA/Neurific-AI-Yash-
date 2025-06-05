import sys
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass  # Fallback to system sqlite3 if pysqlite3-binary is not installed

import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
import os
import logging
from datetime import datetime, timedelta
import time
import re
from langchain.schema import Document
import sqlite3
from langdetect import detect, DetectorFactory
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import numpy as np
import torch  # Explicitly import torch for device control

# Ensure deterministic language detection
DetectorFactory.seed = 0

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Gemini client
try:
    genai.configure(api_key="AIzaSyDSdx_r2bsGNxDydqUAxXDeyL795YONMnc")  # Use .env for API key
    model = genai.GenerativeModel("gemini-1.5-pro")  # Updated to a valid model
except Exception as e:
    st.error(f"Failed to initialize Gemini: {e}")
    logger.error(f"Gemini initialization error: {e}")
    st.stop()

# Initialize vector store
try:
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}  # Explicitly set device to CPU
    )
    vectorstore = Chroma(
        collection_name="lstm_rag",
        embedding_function=embedding_model,
        persist_directory="/tmp/chroma_db"  # Use /tmp for Streamlit Cloud
    )
except Exception as e:
    st.error(f"Failed to initialize vector store: {e}")
    logger.error(f"Vector store initialization error: {e}")
    st.stop()

# Populate vector store with sample data
def populate_vectorstore():
    try:
        # Check if collection is empty
        collection = vectorstore._client.get_collection("lstm_rag")
        if collection.count() > 0:
            logger.info("Vector store already populated")
            return

        # Sample data from Understanding LSTMs and CMU LSTM Notes
        documents = [
            Document(
                page_content="LSTM (Long Short-Term Memory) is a type of RNN designed to model long-term dependencies. It addresses the vanishing gradient problem in standard RNNs by introducing gates: forget, input, and output.",
                metadata={"source": "Understanding LSTMs", "section": "Introduction", "page_number": 1, "content_type": "text", "source_id": "olah_1"}
            ),
            Document(
                page_content="The LSTM architecture includes a cell state and three gates: forget gate, input gate, and output gate. The forget gate decides what information to discard from the cell state.",
                metadata={"source": "CMU LSTM Notes", "section": "Architecture", "page_number": 5, "content_type": "text", "source_id": "cmu_5"}
            ),
            Document(
                page_content="f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)",
                metadata={"source": "CMU LSTM Notes", "section": "Equations", "page_number": 6, "content_type": "equation", "raw_content": "f_t = \\sigma(W_f \\cdot [h_{t-1}, x_t] + b_f)", "source_id": "cmu_6"}
            ),
            Document(
                page_content="Diagram of LSTM cell showing forget, input, and output gates with cell state flow.",
                metadata={
                    "source": "Understanding LSTMs",
                    "section": "Diagram",
                    "page_number": 3,
                    "content_type": "image",
                    "raw_content": "/mount/src/neurific-ai-yash-/images/lstm_diagram.png",
                    "tags": "lstm, diagram, architecture, gates",
                    "source_id": "olah_3"
                }
            )
        ]

        # Add documents to vector store
        vectorstore.add_documents(documents)
        logger.info("Populated vector store with sample data")
    except Exception as e:
        logger.error(f"Error populating vector store: {e}")
        st.error(f"Failed to populate vector store: {e}")

# Call populate_vectorstore after initialization
populate_vectorstore()

# Initialize sentence transformer
intent_embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')  # Explicitly set device to CPU

# SQLite for chat history
def init_db():
    try:
        conn = sqlite3.connect('/tmp/chat_history.db')  # Use /tmp for Streamlit Cloud
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS chats
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      session_id TEXT,
                      timestamp TEXT,
                      role TEXT NOT NULL,
                      content TEXT NOT NULL,
                      images TEXT,
                      language TEXT)''')
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to initialize SQLite database: {e}")
    finally:
        conn.close()

def save_message(session_id, role, content, images, language):
    try:
        conn = sqlite3.connect('/tmp/chat_history.db')  # Use /tmp for Streamlit Cloud
        c = conn.cursor()
        images_str = ','.join([img['path'] for img in images]) if images else ''
        c.execute("INSERT INTO chats (session_id, timestamp, role, content, images, language) VALUES (?, ?, ?, ?, ?, ?)",
                  (session_id, datetime.now().isoformat(), role, content, images_str, language))
        conn.commit()
    except Exception as e:
        logger.error(f"Error saving message to SQLite: {e}")
    finally:
        conn.close()

def get_sessions():
    try:
        conn = sqlite3.connect('/tmp/chat_history.db')  # Use /tmp for Streamlit Cloud
        c = conn.cursor()
        c.execute("SELECT DISTINCT session_id, MAX(timestamp) FROM chats GROUP BY session_id ORDER BY timestamp DESC")
        sessions = c.fetchall()
        return sessions
    except Exception as e:
        logger.error(f"Error retrieving sessions: {e}")
        return []
    finally:
        conn.close()

def get_session_messages(session_id):
    try:
        conn = sqlite3.connect('/tmp/chat_history.db')  # Use /tmp for Streamlit Cloud
        c = conn.cursor()
        c.execute("SELECT role, content, images, timestamp, language FROM chats WHERE session_id = ? ORDER BY timestamp", (session_id,))
        messages = c.fetchall()
        return messages
    except Exception as e:
        logger.error(f"Error retrieving session messages: {e}")
        return []
    finally:
        conn.close()

init_db()

# Rate limiting for Gemini (30 RPM)
request_times = []
def rate_limit():
    global request_times
    current_time = datetime.now()
    request_times = [t for t in request_times if t > current_time - timedelta(minutes=1)]
    if len(request_times) >= 30:
        wait_time = 60 - (current_time - request_times[0]).seconds
        logger.warning(f"Rate limit reached. Waiting {wait_time} seconds.")
        time.sleep(wait_time)
    request_times.append(current_time)

# Clean LaTeX
def clean_latex(text):
    try:
        if not text or text.strip() in ['=', '', '$$', ' ']:
            logger.debug(f"Skipping empty or invalid LaTeX content: {text}")
            return None
        text = text.replace('\\\\', '\\').strip()
        text = re.sub(r'\${2,}', '$', text)
        text = text.strip('$').strip()
        replacements = [
            (r'\s*~\s*([a-zA-Z])', r'\\tilde{\1}'),
            (r"C'\s*_t", r'\\tilde{C}_t'),
            (r'\[([^\]]+),\s*([^\]]+)\]', r'[\1, \2]'),
            (r'(\w+)\s*\[\s*([^\]]+)\]', r'\1 \\cdot [\2]'),
            (r'\\begin{cases}(.*?)\\end{cases}', r'\\begin{cases}\1\\end{cases}', re.DOTALL),
            (r'\\begin{align\*}(.*?)\\end{align\*}', r'\\begin{align*}\1\\end{align*}', re.DOTALL),
            (r'\\begin{align}(.*?)\\end{align}', r'\\begin{align*}\1\\end{align*}', re.DOTALL)
        ]
        for pattern, repl in replacements:
            text = re.sub(pattern, repl, text)
        text = text.replace('cdot', '\\cdot').replace('tanh', '\\tanh').replace('sigma', '\\sigma')
        if not text or not (text.startswith('\\') or text[0].isalnum() or text[0] in '[{('):
            logger.debug(f"Invalid LaTeX start: {text}")
            return None
        return f"$${text}$$"
    except Exception as e:
        logger.error(f"Error cleaning LaTeX: {e}, Input: {text}")
        return None

# Validate LaTeX
def validate_latex(text):
    try:
        if not text:
            logger.debug("Invalid LaTeX: empty input")
            return False
        stripped = text.strip('$').strip()
        if not stripped or stripped in ['=', '', '{}']:
            logger.debug(f"Invalid LaTeX: trivial content {text}")
            return False
        if not re.search(r'\\[a-zA-Z]+|[+\-*/=∑∫∆]', stripped):
            logger.debug(f"Invalid LaTeX: no valid commands or operators {text}")
            return False
        brace_count = 0
        for char in stripped:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
            if brace_count < 0:
                logger.debug(f"Invalid LaTeX: unbalanced braces {text}")
                return False
        if brace_count != 0:
            logger.debug(f"Invalid LaTeX: unbalanced braces {text}")
            return False
        return True
    except Exception as e:
        logger.error(f"LaTeX validation error: {e}, Input: {text}")
        return False

# Format equation block
def format_equation_block(equation, description, source, section, page, score, source_id):
    try:
        if not validate_latex(equation):
            return f"**Invalid Equation**: {description}\n**Source**: {source}, **Section**: {section}, **Page**: {page}, **Score**: {score:.4f}\n"
        return (
            f"**Equation**:\n{equation}\n"
            f"**Description**: {description}\n"
            f"**Source**: {source}, **Section**: {section}, **Page**: {page}, **Similarity Score**: {score:.4f}, **Source ID**: {source_id}\n"
        )
    except Exception as e:
        logger.error(f"Error formatting equation: {e}")
        return f"**Error in Equation**: {description}\n"

# Detect language
def detect_language(text):
    try:
        lang = detect(text)
        logger.info(f"Detected language: {lang} for query: {text}")
        return lang
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        return 'en'

# Detect query intent
def detect_query_intent(query):
    try:
        query_embedding = intent_embedder.encode([query])[0]
        intents = {
            'summary': ['what is', 'define', 'explain simply', 'basic', 'overview', 'in simple terms'],
            'technical': ['how', 'technical', 'equation', 'math', 'details', 'architecture', 'mechanism'],
            'step_by_step': ['steps', 'process', 'walk through', 'how does', 'procedure', 'step-by-step'],
            'table': ['table', 'tabular', 'compare', 'advantages', 'disadvantages', 'summary table'],
            'visual': ['image', 'diagram', 'visual', 'picture', 'figure', 'illustration', 'show images']
        }
        intent_scores = {}
        for intent, keywords in intents.items():
            keyword_embeddings = intent_embedder.encode(keywords)
            scores = [np.dot(query_embedding, ke) / (np.linalg.norm(query_embedding) * np.linalg.norm(ke)) for ke in keyword_embeddings]
            intent_scores[intent] = max(scores)
        intent = max(intent_scores, key=intent_scores.get)
        logger.info(f"Detected intent: {intent} for query: {query}")
        return intent
    except Exception as e:
        logger.error(f"Intent detection error: {e}")
        return 'technical'

# Check if query is LSTM-related
def is_lstm_related(query):
    lstm_keywords = {
        'lstm', 'lstms', 'long short-term memory', 'long short term memory', 'rnn', 'rnns', 'recurrent neural network',
        'gate', 'cell', 'state', 'forget', 'input', 'output', 'sigmoid', 'tanh', 'recurrent', 'network', 'neural',
        'memory', 'architecture', 'diagram', 'sequence', 'dependency', 'gradient', 'vanishing'
    }
    query_words = set(re.split(r'\W+', query.lower()))
    return bool(query_words & lstm_keywords)

# Generate response
@st.cache_data(ttl=3600)
def generate_response(query, session_id, lang='en'):
    try:
        # Check if query is LSTM-related
        if not is_lstm_related(query):
            logger.warning(f"Query not related to LSTMs: {query}")
            return "Sorry, I can only answer questions about Long Short-Term Memory (LSTM) networks or related topics like RNNs based on 'Understanding LSTMs' by Chris Olah and 'CMU LSTM Notes.' Please ask an LSTM-related question.", []

        intent_type = detect_query_intent(query)
        if lang not in ['en', 'hi', 'es', 'fr']:
            lang = 'en'
        logger.info(f"Processing query: {query}, intent: {intent_type}, language: {lang}")

        # Retrieve chunks
        results = vectorstore.similarity_search_with_score(query, k=50)
        logger.info(f"Retrieved {len(results)} chunks for query: {query}")
        for doc, score in results[:5]:  # Log top 5 results
            logger.info(f"Chunk: {doc.page_content[:100]}..., Score: {score}, Metadata: {doc.metadata}")

        if not results:
            logger.warning(f"No relevant chunks found for query: {query}")
            return "No relevant information found in the database. Please try a different LSTM-related question.", []

        # Prepare context, images, and references
        context = []
        images = []
        chunk_count = 0
        total_length = 0
        max_context_length = 10000
        source_references = {}

        query_keywords = set(re.split(r'\W+', query.lower()))
        relevant_keywords = {'lstm', 'gate', 'cell', 'state', 'forget', 'input', 'output', 'diagram',
                            'architecture', 'network', 'memory', 'sigmoid', 'tanh', 'flow', 'structure', 'rnn', 'recurrent'}

        for doc, score in results:
            content = doc.page_content
            metadata = doc.metadata
            source = metadata.get('source', 'Unknown')
            source_id = metadata.get('source_id', 'Unknown')
            section = metadata.get('section', 'Unknown')
            page = metadata.get('page_number', 'N/A')
            content_type = metadata.get('content_type', 'text')

            ref_key = f"{source}, {section}, Page {page}, Source ID: {source_id}"
            source_references[ref_key] = source_references.get(ref_key, 0) + 1

            chunk_text = ""
            if content_type == 'text':
                chunk_text = f"**Source**: {source}, **Section**: {section}, **Page**: {page}, **Source ID**: {source_id}\n**Content**: {content}\n**Similarity Score**: {score:.4f}"
            elif content_type == 'equation':
                raw_latex = metadata.get('raw_content', content)
                cleaned_latex = clean_latex(raw_latex)
                if cleaned_latex and validate_latex(cleaned_latex):
                    chunk_text = format_equation_block(cleaned_latex, content, source, section, page, score, source_id)
                else:
                    logger.debug(f"Skipping invalid equation: {raw_latex}")
                    continue
            elif content_type == 'image':
                tags = metadata.get('tags', '').split(',') if metadata.get('tags') else []
                chunk_text = f"**Source**: {source}, **Section**: {section}, **Page**: {page}, **Source ID**: {source_id}\n**Image Description**: {content}\n**Similarity Score**: {score:.4f}\n**Tags**: {metadata.get('tags', '')}"
                img_path = metadata.get('raw_content', '')
                img_path = os.path.normpath(os.path.abspath(img_path))
                if os.path.exists(img_path):
                    content_keywords = set(content.lower().split())
                    tag_match = bool(set(tags) & (query_keywords | relevant_keywords))
                    keyword_match = bool((query_keywords | relevant_keywords) & content_keywords)
                    if (intent_type == 'visual' or 'diagram' in query.lower() or tag_match or score > 0.2) and (keyword_match or tag_match or score > 0.3):
                        images.append({
                            'path': img_path,
                            'caption': f"Image from {source}, Section: {section}, Page: {page}, Source ID: {source_id} - {content}",
                            'score': float(score),
                            'metadata': metadata,
                            'tags': tags
                        })
                        logger.info(f"Selected image: {img_path}, Score: {score:.4f}, Tags: {tags}")
                    else:
                        logger.debug(f"Skipped image: {img_path}, Low relevance (Score: {score:.4f})")
                else:
                    logger.warning(f"Image path not found: {img_path}")

            if chunk_text and total_length + len(chunk_text) <= max_context_length:
                context.append(chunk_text)
                total_length += len(chunk_text)
                chunk_count += 1

        # Fallback for visual intent
        if not images and intent_type == 'visual':
            for doc, score in results:
                if doc.metadata.get('content_type') == 'image':
                    img_path = doc.metadata.get('raw_content', '')
                    source_id = doc.metadata.get('source_id', 'Unknown')
                    tags = doc.metadata.get('tags', '').split(',') if doc.metadata.get('tags') else []
                    if os.path.exists(img_path) and (score > 0.15 or bool(set(tags) & (query_keywords | relevant_keywords))):
                        images.append({
                            'path': img_path,
                            'caption': f"Image from {doc.metadata.get('source', 'Unknown')}, Section: {doc.metadata.get('section', 'Unknown')}, Page: {doc.metadata.get('page_number', 'N/A')}, Source ID: {source_id} - {doc.page_content}",
                            'score': float(score),
                            'metadata': doc.metadata,
                            'tags': tags
                        })
                        ref_key = f"{doc.metadata.get('source', 'Unknown')}, {doc.metadata.get('section', 'Unknown')}, Page {doc.metadata.get('page_number', 'N/A')}, Source ID: {source_id}"
                        source_references[ref_key] = source_references.get(ref_key, 0) + 1
                        logger.info(f"Fallback image selected: {img_path}, Score: {score:.4f}")
                        if len(images) >= 3:
                            break

        logger.info(f"Retrieved {chunk_count} chunks and {len(images)} images for query: {query}")

        # Chat history context
        messages = get_session_messages(session_id)
        history_context = "\n".join([f"{m[0]} ({m[4]}): {m[1]}" for m in messages[-5:]])

        # Prompt
        context_str = '\n\n'.join(context)
        knowledge_base_summary = (
            "You are an advanced AI assistant specializing in Long Short-Term Memory (LSTM) networks and Recurrent Neural Networks (RNNs). "
            "Your knowledge is derived from two primary sources: 'Understanding LSTMs' by Chris Olah (a blog post) and 'CMU LSTM Notes' (Spring 2023, a PDF document). "
            "These sources cover LSTM architecture, equations, advantages, limitations, and comparisons with RNNs, including diagrams and experimental results. "
            "You can provide responses in multiple languages (English, Hindi, Spanish, French), render LaTeX equations, and describe relevant images with detailed explanations. "
            "Your responses are grounded in the provided sources, and you can detect query intent to tailor responses (e.g., summary, technical, step-by-step, tabular, or visual). "
            "For queries outside the scope of LSTMs or RNNs, you politely redirect users to ask relevant questions. "
            "When handling visual queries, prioritize images with tags or descriptions matching the query and provide in-depth explanations of their content and relevance."
        )

        if intent_type == "summary":
            response_instruction = (
                f"- Provide a concise summary (100-150 words) in a single paragraph, in {lang}, suitable for beginners.\n"
                "- Use simple, conversational language and avoid technical jargon.\n"
                "- Include a brief description of one key image if available, explaining its relevance.\n"
                "- Render equations in LaTeX within $$ delimiters, using correct syntax.\n"
                "- If no valid equations, explain the concept in plain text.\n"
            )
        elif intent_type == "step_by_step":
            response_instruction = (
                f"- Provide a step-by-step explanation in {lang} with numbered steps (1., 2., etc.).\n"
                "- Use a narrative tone with clear transitions.\n"
                "- Include relevant equations in LaTeX within $$ delimiters and describe one image if available.\n"
                "- If no valid equations, use text-based explanations.\n"
            )
        elif intent_type == "table":
            response_instruction = (
                f"- Provide a tabular explanation in {lang} using markdown table syntax.\n"
                "- Include columns for relevant categories (e.g., Component, Description, Source, Section, Page).\n"
                "- Include equations in LaTeX in a separate section if available.\n"
                "- Describe one image if available.\n"
            )
        elif intent_type == "visual":
            response_instruction = (
                f"- Provide a detailed explanation in {lang} focusing on visual elements, including at least one relevant diagram.\n"
                "- For each image, describe:\n"
                "  - The visual content (e.g., components, labels, flow).\n"
                "  - How it relates to the query and LSTMs.\n"
                "  - The source, section, page, and source ID.\n"
                "- Use bullet points for clarity.\n"
                "- Include relevant equations in LaTeX within $$ delimiters if applicable.\n"
                "- If no relevant images are found, explain why and suggest related concepts.\n"
                "- Cite sources in markdown format.\n"
            )
        else:  # technical
            response_instruction = (
                f"- Provide a detailed technical explanation in {lang} with sections: Overview, Equations, Explanations, Images.\n"
                "- Use bullet points for clarity.\n"
                "- Include all relevant equations in LaTeX within $$ delimiters, using \\begin{align*} for multi-line equations.\n"
                "- Explain each equation’s variables and role in LSTMs.\n"
                "- Describe all relevant images with detailed explanations of their content and relevance.\n"
                "- If no valid equations, provide a detailed text-based explanation.\n"
                "- Cite sources with sections, pages, and source IDs in markdown format.\n"
            )

        prompt = (
            f"{knowledge_base_summary}\n\n"
            f"Answer the query: \"{query}\" in the language {lang}, ensuring the response is clear, accurate, and strictly based on the provided sources.\n\n"
            f"Recent Chat History:\n{history_context}\n\n"
            f"Context:\n{context_str}\n\n"
            f"Instructions:\n{response_instruction}\n"
            "- Directly quote relevant chunks in markdown blockquotes, citing source, section, page, and source ID.\n"
            "- If no relevant images are available, state explicitly (e.g., 'No relevant images found') and explain why.\n"
            "- Use a professional, academic tone unless a summary is requested, then use a conversational tone.\n"
            "- Ensure all LaTeX equations are formatted within $$ delimiters with correct syntax (e.g., \\tilde{C}_t, \\cdot, \\sigma, \\tanh). Use \\begin{cases} for piecewise equations and \\begin{align*} for multi-line equations.\n"
            "- Use markdown for clear formatting: ## for sections, - for lists, | for tables, two newlines between sections.\n"
            "- Include a 'References' section listing unique sources, sections, pages, and source IDs used, without repetition.\n"
            "- Avoid duplicating references; list each unique source-section-page combination only once.\n"
        )

        # Generate response
        rate_limit()
        response = model.generate_content(prompt)
        response_text = response.text + "\n\n## References\n" + "\n".join([f"- {k}" for k, v in source_references.items()])
        logger.info(f"Generated response for query: {query}")
        return response_text, images
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"Error generating response: {e}", []

# Streamlit UI
st.set_page_config(page_title="Advanced LSTM Chatbot", layout="wide")
st.title("Advanced LSTM Chatbot")
st.markdown("Ask questions about LSTMs or RNNs based on *Understanding LSTMs* by Chris Olah and *CMU LSTM Notes* (Spring 2023). Supports multilingual responses, equations, and images. Powered by Google Gemini.")

# Add MathJax 3 with explicit rendering
st.markdown("""
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']],
                processEscapes: true,
                processEnvironments: true
            },
            options: {
                skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
                ignoreHtmlClass: 'tex2jax_ignore'
            },
            startup: {
                ready: function() {
                    MathJax.startup.defaultReady();
                    MathJax.startup.promise.then(() => {
                        console.log('MathJax is loaded and ready');
                    });
                }
            }
        };
        async function renderMathJax() {
            if (window.MathJax && typeof MathJax.typesetPromise === 'function') {
                await MathJax.typesetPromise();
            }
        }
    </script>
""", unsafe_allow_html=True)

# Language selection
language_options = {
    'English': 'en',
    'Hindi': 'hi',
    'Spanish': 'es',
    'French': 'fr'
}
selected_language = st.sidebar.selectbox("Select Response Language", list(language_options.keys()), index=0)
preferred_lang = language_options[selected_language]

# Session management
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(datetime.now().timestamp())
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Session selector
sessions = get_sessions()
session_options = {f"Session {i+1} ({s[1][:19]})": s[0] for i, s in enumerate(sessions)}
session_options['New Session'] = 'new'
selected_session = st.sidebar.selectbox("Select Chat Session", list(session_options.keys()))
if selected_session == 'New Session':
    st.session_state.session_id = str(datetime.now().timestamp())
    st.session_state.messages = []
else:
    st.session_state.session_id = session_options[selected_session]
    st.session_state.messages = []
    for role, content, images_str, _, lang in get_session_messages(st.session_state.session_id):
        images = [{'path': p, 'caption': 'Image'} for p in images_str.split(',') if p]
        st.session_state.messages.append({'role': role, 'content': content, 'images': images, 'language': lang})

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(f"**Language**: {message.get('language', 'en')}\n\n{message['content']}", unsafe_allow_html=True)
        if "images" in message and message["images"]:
            for img in message["images"]:
                if os.path.exists(img["path"]):
                    st.image(img["path"], caption=img.get("caption", "Image"))
                else:
                    st.warning(f"Image not found: {img['path']}")
                    logger.warning(f"Image not found in chat history: {img['path']}")

# User input
if query := st.chat_input("Ask a question about LSTMs or RNNs"):
    # Use user-selected language
    lang = preferred_lang
    if re.search(r'\b(in|explain in)\s+(hindi|spanish|french|english)\b', query.lower()):
        lang_map = {'hindi': 'hi', 'spanish': 'es', 'french': 'fr', 'english': 'en'}
        for lang_name, lang_code in lang_map.items():
            if lang_name in query.lower():
                lang = lang_code
                break
    else:
        detected_lang = detect_language(query)
        if detected_lang in language_options.values():
            lang = detected_lang

    # Add user message
    st.session_state.messages.append({"role": "user", "content": query, "language": lang})
    save_message(st.session_state.session_id, "user", query, [], lang)
    with st.chat_message("user"):
        st.markdown(f"**Language**: {lang}\n\n{query}")

    # Generate and stream response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_container = st.empty()
            response_text, images = generate_response(query, st.session_state.session_id, lang)

        # Stream response
        streamed_text = ""
        sections = []
        current_section = ""
        equation_blocks = []
        in_table = False

        lines = response_text.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('## '):
                if current_section or equation_blocks:
                    sections.append((current_section, equation_blocks))
                    equation_blocks = []
                current_section = line + '\n\n'
            elif line.startswith('|'):
                in_table = True
                current_section += line + '\n'
            elif in_table and not line:
                in_table = False
                current_section += line + '\n'
            elif line.startswith('$$') and not line.endswith('$$'):
                eq_lines = [line[2:]]
                i += 1
                while i < len(lines) and not lines[i].strip().endswith('$$'):
                    eq_lines.append(lines[i].strip())
                    i += 1
                if i < len(lines) and lines[i].strip().endswith('$$'):
                    eq_lines.append(lines[i].strip()[:-2])
                eq_text = ' '.join(eq_lines).strip()
                cleaned = clean_latex(eq_text)
                if cleaned and validate_latex(cleaned):
                    equation_blocks.append(cleaned.strip('$'))
                else:
                    logger.debug(f"Skipping invalid LaTeX: {eq_text}")
                    current_section += f"$$ {eq_text} $$\n"
            elif line.startswith('$$') and line.endswith('$$'):
                cleaned = clean_latex(line.strip('$'))
                if cleaned and validate_latex(cleaned):
                    equation_blocks.append(cleaned.strip('$'))
                else:
                    logger.debug(f"Skipping invalid LaTeX: {line}")
                    current_section += line + '\n'
            else:
                current_section += line + '\n'
            i += 1
        if current_section or equation_blocks:
            sections.append((current_section, equation_blocks))

        for section, eq_blocks in sections:
            streamed_text += section
            response_container.markdown(streamed_text, unsafe_allow_html=True)
            time.sleep(0.05)
            for eq in eq_blocks:
                eq_div = f"\n<div class='math-equation'>$${eq}$$</div>\n"
                streamed_text += eq_div
                response_container.markdown(streamed_text, unsafe_allow_html=True)
                st.markdown("<script>renderMathJax();</script>", unsafe_allow_html=True)
                time.sleep(0.1)

        # Final MathJax rendering
        st.markdown("<script>renderMathJax();</script>", unsafe_allow_html=True)

        # Display images with explanations
        if images:
            st.subheader("Relevant Images")
            for img in sorted(images, key=lambda x: x['score'], reverse=True)[:3]:
                if os.path.exists(img["path"]):
                    st.image(img["path"], caption=img["caption"])
                    explanation = (
                        f"""
                        **Image Explanation**:\n
                        - **Content**: This image likely describes {img['metadata'].get('tags', 'LSTM components')}, as described in {img['metadata'].get('source', 'Unknown')}, Section: {img['metadata'].get('section', 'Unknown')}.\n
                        - **Relevance**: It relates to the query '{query}' by illustrating {', '.join(img['tags']) if img['tags'] else 'LSTM concepts'}.\n
                        - **Source**: {img['metadata'].get('source', 'Unknown')}, Page: {img['metadata'].get('page_number', 'N/A')}, Source ID: {img['metadata'].get('source_id', 'Unknown')}.\n
                        """
                    )
                    st.markdown(explanation)
                    logger.info(f"Displayed image: {img['path']}, Score: {img['score']:.4f}")
                else:
                    st.warning(f"Image not found: {img['path']}")
                    logger.warning(f"Image not found: {img['path']}")
        else:
            st.info("No relevant images found for this query. Try asking for specific diagrams, e.g., 'show LSTM architecture diagram'.")
            logger.info("No images found for this query.")

        # Save response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "images": images,
            "language": lang
        })
        save_message(st.session_state.session_id, "assistant", response_text, images, lang)

# Footer
st.markdown("---")
st.markdown("<p>Built with Streamlit and Google Gemini. Data sourced from *Understanding LSTMs* by Chris Olah and *CMU LSTM Notes* (Spring 2023).</p>", unsafe_allow_html=True)