#!/usr/bin/env python3
"""
Comprehensive verification script for RAG system
"""
import sys
import os

def test_python_version():
    """Check Python version"""
    print(f"🐍 Python version: {sys.version}")
    if sys.version_info >= (3, 8):
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python version too old (need 3.8+)")
        return False

def test_required_packages():
    """Test if all required packages can be imported"""
    packages = [
        ('beautifulsoup4', 'bs4'),
        ('requests', 'requests'),
        ('numpy', 'numpy'),
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('sentence-transformers', 'sentence_transformers'),
        ('faiss-cpu', 'faiss'),
        ('langchain', 'langchain')
    ]
    
    failed = []
    for package_name, import_name in packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name} imported successfully")
        except ImportError as e:
            print(f"❌ {package_name} import failed: {e}")
            failed.append(package_name)
    
    return len(failed) == 0

def test_document_file():
    """Test document file exists and has content"""
    try:
        with open('Selected_Document.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if len(content) > 100:  # Minimum reasonable length
            print(f"✅ Document file loaded ({len(content)} characters)")
            print(f"📝 First 100 chars: {content[:100]}...")
            return True
        else:
            print("❌ Document file is too short or empty")
            return False
    except FileNotFoundError:
        print("❌ Selected_Document.txt not found")
        return False
    except Exception as e:
        print(f"❌ Error reading document: {e}")
        return False

def test_text_splitting():
    """Test text splitting functionality"""
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Load document
        with open('Selected_Document.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Create splitter
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', ' ', ''],
            chunk_size=500,
            chunk_overlap=50
        )
        
        # Split text
        chunks = text_splitter.split_text(text)
        
        if len(chunks) > 0:
            print(f"✅ Text splitting successful ({len(chunks)} chunks)")
            print(f"📊 Average chunk size: {sum(len(chunk) for chunk in chunks) / len(chunks):.1f} chars")
            return True
        else:
            print("❌ No chunks created")
            return False
            
    except Exception as e:
        print(f"❌ Text splitting failed: {e}")
        return False

def test_embeddings():
    """Test embedding model loading and encoding"""
    try:
        from sentence_transformers import SentenceTransformer
        
        print("⏳ Loading embedding model...")
        model = SentenceTransformer("sentence-transformers/all-distilroberta-v1")
        
        # Test encoding
        test_text = ["This is a test sentence.", "Another test sentence."]
        embeddings = model.encode(test_text)
        
        print(f"✅ Embedding model loaded successfully")
        print(f"📐 Embedding dimension: {embeddings.shape[1]}")
        print(f"🔢 Test embeddings shape: {embeddings.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Embedding model failed: {e}")
        return False

def test_faiss():
    """Test FAISS indexing"""
    try:
        import faiss
        import numpy as np
        
        # Create test data
        dimension = 768
        test_vectors = np.random.random((10, dimension)).astype('float32')
        
        # Create and populate index
        index = faiss.IndexFlatL2(dimension)
        index.add(test_vectors)
        
        # Test search
        query = test_vectors[:1]  # Use first vector as query
        distances, indices = index.search(query, 3)
        
        print(f"✅ FAISS indexing successful")
        print(f"🔍 Index contains {index.ntotal} vectors")
        print(f"📏 Search returned {len(indices[0])} results")
        return True
        
    except Exception as e:
        print(f"❌ FAISS indexing failed: {e}")
        return False

def test_text_generation():
    """Test text generation model"""
    try:
        from transformers import pipeline
        
        print("⏳ Loading text generation model...")
        generator = pipeline('text2text-generation', model='google/flan-t5-small', device=-1)
        
        # Test generation
        test_prompt = "Question: What is AI? Answer:"
        result = generator(test_prompt, max_length=50, num_return_sequences=1)
        
        print(f"✅ Text generation model loaded successfully")
        print(f"🤖 Test generation: {result[0]['generated_text']}")
        return True
        
    except Exception as e:
        print(f"❌ Text generation failed: {e}")
        return False

def test_full_rag_pipeline():
    """Test the complete RAG pipeline"""
    try:
        print("⏳ Testing complete RAG pipeline...")
        
        # This is a simplified version of the main components
        from sentence_transformers import SentenceTransformer
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from transformers import pipeline
        import faiss
        import numpy as np
        
        # Load document
        with open('Selected_Document.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', ' ', ''],
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_text(text)
        
        # Create embeddings (just first 3 chunks for speed)
        model = SentenceTransformer("sentence-transformers/all-distilroberta-v1")
        test_chunks = chunks[:3]  # Use fewer chunks for speed
        embeddings = model.encode(test_chunks).astype('float32')
        
        # Create FAISS index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        
        # Test retrieval
        query = "What is artificial intelligence?"
        query_embedding = model.encode([query]).astype('float32')
        distances, indices = index.search(query_embedding, 2)
        
        # Test generation
        generator = pipeline('text2text-generation', model='google/flan-t5-small', device=-1)
        context = " ".join([test_chunks[i] for i in indices[0]])
        prompt = f"Context: {context[:200]}... Question: {query} Answer:"
        result = generator(prompt, max_length=100)
        
        print(f"✅ Full RAG pipeline test successful")
        print(f"🎯 Retrieved {len(indices[0])} relevant chunks")
        print(f"💭 Generated answer: {result[0]['generated_text']}")
        return True
        
    except Exception as e:
        print(f"❌ Full RAG pipeline failed: {e}")
        return False

def main():
    print("🔍 COMPREHENSIVE RAG SYSTEM VERIFICATION")
    print("=" * 50)
    
    tests = [
        ("Python Version", test_python_version),
        ("Required Packages", test_required_packages),
        ("Document File", test_document_file),
        ("Text Splitting", test_text_splitting),
        ("Embeddings", test_embeddings),
        ("FAISS Indexing", test_faiss),
        ("Text Generation", test_text_generation),
        ("Full RAG Pipeline", test_full_rag_pipeline)
    ]
    
    passed = 0
    failed_tests = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing: {test_name}")
        print("-" * 30)
        try:
            if test_func():
                passed += 1
            else:
                failed_tests.append(test_name)
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            failed_tests.append(test_name)
    
    print(f"\n{'='*50}")
    print(f"📊 FINAL RESULTS: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 ALL TESTS PASSED! RAG system is fully functional.")
        print("\n✨ Your RAG system is ready for:")
        print("   • Interactive question answering")
        print("   • Document analysis")
        print("   • Submission to GitHub")
    else:
        print(f"⚠️  {len(failed_tests)} test(s) failed:")
        for test in failed_tests:
            print(f"   • {test}")
        print("\n🔧 Please address the failed tests before proceeding.")

if __name__ == "__main__":
    main()
