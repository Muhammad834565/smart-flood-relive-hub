import os
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from pypdf import PdfReader
import download_pdf as py_pdf

load_dotenv()
# Load HuggingFace embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Init Chroma DB with persistent storage
chroma_db_path = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=chroma_db_path)
collection = chroma_client.get_or_create_collection(name="docs")

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# FastAPI app
app = FastAPI(title="Mini RAG App with PDF Support + Doc Filtering")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],  # Angular dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Helper: Extract text from PDF
def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# 1. Upload PDF & store embeddings (with auto-generated doc_id)
def upload_document(file_path: str):
    if not file_path.endswith(".pdf"):
        return {"error": "Only PDF files are supported"}

    # Generate unique document ID
    doc_id = str(uuid.uuid4())
    doc_id = "a"
    text = extract_text_from_pdf(file_path)
    
    chunk_size = 1000
    overlap = 200
    chunks = []
    
    # Process the entire document text
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk.strip())
    
    # Ensure we process ALL chunks from the document
    print(f"Processing {len(chunks)} chunks from document: {file_path}")
    
    # Store each chunk in the database
    for idx, chunk in enumerate(chunks):
        try:
            emb = embedding_model.encode(chunk).tolist()
            collection.add(
                ids=[f"{doc_id}_{idx}"],
                embeddings=[emb],
                documents=[chunk],
                metadatas=[{"doc_id": doc_id, "filename": file_path, "chunk_index": idx}]
            )
            print(f"Stored chunk {idx + 1}/{len(chunks)}")
        except Exception as e:
            print(f"Error storing chunk {idx}: {e}")
            continue

    return {print(f"Successfully stored ALL {len(chunks)} chunks from {file_path}\ndoc_id: {doc_id}\ntotal_chunks: {len(chunks)}\ndocument_length: {len(text)}")    }

# Example usage
file = "downloads/a.pdf"
result = upload_document(file)
print(result)



# 2. Ask question (RAG with doc_id filter)
@app.post("/ask")
async def ask_question(question: str = Form(...)):
    doc_id = "a"  # For testing, use the known doc_id
    q_emb = embedding_model.encode(question).tolist()

    # Retrieve more chunks for better context
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=5,  # Increased from 3 to 5 for more comprehensive results
        where={"doc_id": doc_id}  # filter by doc_id
    )

    if not results["documents"] or not results["documents"][0]:
        return JSONResponse({"answer": f"No content found for document {doc_id}"})

    context = " ".join(results["documents"][0])

    # Use Gemini 1.5 Flash (more reliable free model)
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    # Enhanced prompt for better document analysis
    prompt = f"""You are a dual-role expert: a **Smart Flood Relief Agent** and a **Document Analyst**.

Your primary goal is to provide comprehensive, actionable, and real-time flood relief guidance, specifically including **phone numbers** and **shelter locations**.

You must first analyze the provided `DOCUMENT CONTENT` to extract the foundational technical and architectural knowledge for building the hub. Then, you will use **Google Search** to find real-world, relevant flood relief data (e.g., sample emergency hotlines, typical shelter organization details) to demonstrate how the system described in the document would function in a live scenario.

***

**DOCUMENT CONTENT:**
{context}

**QUESTION:** {question}

***

**ANALYSIS AND RESPONSE INSTRUCTIONS:**

1.  **Agent Persona (Guidance Focus):** Frame the final answer as the output of the *Smart Flood Relief Hub* itself. The technical details extracted from the document should serve as the *supporting architecture* for the guidance provided.
2.  **Extraction and Synthesis:**
    * **Document Analysis:** Extract all technical and architectural information relevant to building the hub (Angular, FastAPI, Chrome DB/IndexedDB, AI/LLM integration, real-time data methods, etc.).
    * **External Data Integration (Google Search):** Use Google to find **generic, public examples** of flood relief information (e.g., **"national flood emergency hotline number U.S."** or **"sample public flood shelter location data"**) to populate the guidance section.
3.  **Structure and Detail:** Ensure the response has two distinct, well-defined sections:
    * **Section A: Immediate Flood Relief Guidance (The Agent's Output):** This should be the direct answer to the user's need (phone numbers, shelter locations, based on Google search examples).
    * **Section B: System Architecture Overview (The Analyst's Findings):** This should be the summary of the technical information extracted from the `DOCUMENT CONTENT` on *how* the hub provides the guidance.
4.  **Formatting:** Use clear markdown with headers, bullet points, and strong emphasis.

***

**RESPONSE FORMAT:**

# 🚨 Smart Flood Relief Hub: Real-Time Guidance 🚨

*Start with a brief, empathetic direct answer.*

## 📞 Key Relief Contact Information (Sample Data)
*Use Google to provide a generic, relevant example for an emergency contact number.*

## 🏠 Nearest Shelter and Relief Center Locations (Sample Data)
*Use Google to provide an example of typical structured data a shelter system would display (e.g., location name, capacity, status).*

---

# ⚙️ System Architecture: How the Hub Works (Document Analysis)

*This section provides the context and supporting details based on the **DOCUMENT CONTENT**.*

## 🛠️ Extracted Technical Components
*Summarize the Angular, FastAPI, and 'Chrome DB' elements.*

## 🧠 AI Agent and Real-Time Logic
*Detail the AI integration and real-time communication methods (WebSockets, data APIs) mentioned in the document.*

***

**Handle Exceptions:**
- **If the question is not relevant to the document:** **Not Relevant**
- **If insufficient *technical* information is found in the document:** **Insufficient Technical Architecture Information**
- **If no content is available:** **No Content**
"""

    
    response = model.generate_content(prompt)

    return JSONResponse({
        "answer": response.text
    })
