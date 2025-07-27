import logging
import warnings
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Suppress noisy logs with error handling
try:
    import transformers.logging as hf_logging
    hf_logging.set_verbosity_error()
except ImportError:
    print("Note: transformers.logging not available, using basic logging")

logging.getLogger('langchain.text_splitter').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

# Parameters
chunk_size = 500
chunk_overlap = 50
model_name = "sentence-transformers/all-distilroberta-v1"
top_k = 5

print("🚀 RAG System Starting...")
print("=" * 30)

# Read the pre-scraped document
print("📄 Loading document...")
try:
    with open('Selected_Document.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"✅ Document loaded. Length: {len(text)} characters")
except FileNotFoundError:
    print("❌ Selected_Document.txt not found!")
    print("Please run text_extractor.py first or ensure the file exists.")
    exit(1)

# Split into appropriately-sized chunks
print("✂️  Splitting document into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', ' ', ''],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
chunks = text_splitter.split_text(text)
print(f"✅ Created {len(chunks)} chunks")

# Embed & Build FAISS Index
print("🧠 Loading embedding model and creating embeddings...")
try:
    embedder = SentenceTransformer(model_name)
    print("✅ Embedding model loaded")
except Exception as e:
    print(f"❌ Failed to load embedding model: {e}")
    print("💡 Try: pip install sentence-transformers")
    exit(1)

# Encode chunks with progress bar
print("🔢 Creating embeddings...")
try:
    embeddings = embedder.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)
    print(f"✅ Embeddings created with shape {embeddings.shape}")
except Exception as e:
    print(f"❌ Failed to create embeddings: {e}")
    exit(1)

# Initialize FAISS index
print("🗃️  Building FAISS index...")
try:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"✅ FAISS index created with {index.ntotal} vectors of dimension {dimension}")
except Exception as e:
    print(f"❌ Failed to create FAISS index: {e}")
    exit(1)

# Load the Generator Pipeline
print("🤖 Loading text generation model...")
try:
    generator = pipeline('text2text-generation', model='google/flan-t5-small', device=-1)
    print("✅ Generator model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load generator model: {e}")
    print("💡 Try: pip install transformers torch")
    exit(1)

def retrieve_chunks(question, k=top_k):
    """
    Encode the question, search the FAISS index, return top k chunks
    """
    try:
        # Encode the question
        question_embedding = embedder.encode([question], convert_to_numpy=True).astype(np.float32)
        
        # Search the index
        distances, indices = index.search(question_embedding, k)
        
        # Return the top k chunks
        retrieved_chunks = [chunks[i] for i in indices[0]]
        return retrieved_chunks
    except Exception as e:
        print(f"Error in retrieval: {e}")
        return []

def answer_question(question):
    """
    Call retrieve_chunks, build a prompt with context, call generator, and return generated_text
    """
    try:
        # Retrieve relevant chunks
        context_chunks = retrieve_chunks(question)
        
        if not context_chunks:
            return "Sorry, I couldn't find relevant information to answer your question."
        
        # Build context from retrieved chunks
        context = "\n\n".join(context_chunks)
        
        # Create prompt
        prompt = f"""Based on the following context, answer the question:

Context:
{context}

Question: {question}

Answer:"""
        
        # Generate answer
        result = generator(prompt, max_length=150, num_return_sequences=1)
        generated_text = result[0]['generated_text']
        
        return generated_text
    except Exception as e:
        return f"Error generating answer: {e}"

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🎉 RAG System Ready!")
    print("="*50)
    print("Enter 'exit' or 'quit' to end.")
    print("Try asking questions about artificial intelligence!")
    
    # Suggest some example questions
    print("\n💡 Example questions:")
    print("   • What is artificial intelligence?")
    print("   • How does machine learning work?")
    print("   • What are AI applications?")
    print("   • What is natural language processing?")
    
    while True:
        question = input("\n🤔 Your question: ")
        if question.lower() in ("exit", "quit", "q"):
            print("👋 Goodbye!")
            break
        
        if not question.strip():
            print("Please enter a question.")
            continue
        
        try:
            print("🔍 Searching for relevant information...")
            answer = answer_question(question)
            print(f"\n💬 Answer: {answer}")
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
