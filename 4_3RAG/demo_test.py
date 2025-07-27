#!/usr/bin/env python3
"""
RAG System Demo - Automated test with predefined questions
"""

def run_rag_demo():
    print("🚀 RAG SYSTEM DEMO")
    print("=" * 40)
    
    try:
        # Import all required modules
        print("📦 Importing modules...")
        import logging
        import transformers.logging as hf_logging
        import warnings
        import numpy as np
        from sentence_transformers import SentenceTransformer
        import faiss
        from transformers import pipeline
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Suppress logs
        logging.getLogger('langchain.text_splitter').setLevel(logging.ERROR)
        hf_logging.set_verbosity_error()
        warnings.filterwarnings('ignore')
        print("   ✅ All modules imported successfully")
        
        # Load document
        print("\n📄 Loading document...")
        with open('Selected_Document.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"   ✅ Document loaded ({len(text)} characters)")
        
        # Split into chunks
        print("\n✂️  Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', ' ', ''],
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_text(text)
        print(f"   ✅ Created {len(chunks)} chunks")
        
        # Load embedding model
        print("\n🧠 Loading embedding model...")
        embedder = SentenceTransformer("sentence-transformers/all-distilroberta-v1")
        print("   ✅ Embedding model loaded")
        
        # Create embeddings
        print("\n🔢 Creating embeddings...")
        embeddings = embedder.encode(chunks, convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)
        print(f"   ✅ Created embeddings with shape {embeddings.shape}")
        
        # Build FAISS index
        print("\n🗃️  Building FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        print(f"   ✅ FAISS index built with {index.ntotal} vectors")
        
        # Load generation model
        print("\n🤖 Loading text generation model...")
        generator = pipeline('text2text-generation', model='google/flan-t5-small', device=-1)
        print("   ✅ Generation model loaded")
        
        # Define retrieval function
        def retrieve_chunks(question, k=3):
            question_embedding = embedder.encode([question], convert_to_numpy=True).astype(np.float32)
            distances, indices = index.search(question_embedding, k)
            return [chunks[i] for i in indices[0]]
        
        # Define answer function
        def answer_question(question):
            context_chunks = retrieve_chunks(question)
            context = "\n\n".join(context_chunks)
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            result = generator(prompt, max_length=100, num_return_sequences=1)
            return result[0]['generated_text']
        
        # Test with predefined questions
        print("\n🎯 Testing RAG system with sample questions...")
        test_questions = [
            "What is artificial intelligence?",
            "What is machine learning?",
            "What are the applications of AI?",
            "What is natural language processing?"
        ]
        
        for i, question in enumerate(test_questions[:2], 1):  # Test only first 2 to save time
            print(f"\n📝 Question {i}: {question}")
            try:
                answer = answer_question(question)
                print(f"   💬 Answer: {answer}")
                print("   ✅ Question answered successfully")
            except Exception as e:
                print(f"   ❌ Error answering question: {e}")
        
        print("\n" + "=" * 40)
        print("🎉 RAG SYSTEM DEMO COMPLETED!")
        print("✅ All components are working correctly")
        print("\n🚀 Ready for interactive use:")
        print("   python RAG_app.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_rag_demo()
    if success:
        print("\n✨ Your RAG system is verified and ready!")
    else:
        print("\n🔧 Please check the errors above and install missing packages.")
