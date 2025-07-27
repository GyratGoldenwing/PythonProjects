#!/usr/bin/env python3
"""
Simple verification that core RAG components work
"""

def simple_test():
    print("🧪 SIMPLE RAG VERIFICATION")
    print("=" * 30)
    
    # Test 1: Basic imports
    print("1️⃣ Testing basic imports...")
    try:
        import numpy as np
        import logging
        import warnings
        print("   ✅ Basic imports work")
    except Exception as e:
        print(f"   ❌ Basic imports failed: {e}")
        return False
    
    # Test 2: Document loading
    print("2️⃣ Testing document...")
    try:
        with open('Selected_Document.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"   ✅ Document loaded ({len(content)} chars)")
    except Exception as e:
        print(f"   ❌ Document loading failed: {e}")
        return False
    
    # Test 3: Try langchain
    print("3️⃣ Testing text splitting...")
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(content)
        print(f"   ✅ Text split into {len(chunks)} chunks")
    except Exception as e:
        print(f"   ❌ Text splitting failed: {e}")
        return False
    
    # Test 4: Try transformers (basic)
    print("4️⃣ Testing transformers...")
    try:
        import transformers
        print(f"   ✅ Transformers {transformers.__version__} available")
    except Exception as e:
        print(f"   ❌ Transformers not available: {e}")
        return False
    
    # Test 5: Try sentence transformers
    print("5️⃣ Testing sentence transformers...")
    try:
        from sentence_transformers import SentenceTransformer
        print("   ✅ Sentence transformers available")
    except Exception as e:
        print(f"   ❌ Sentence transformers failed: {e}")
        return False
    
    # Test 6: Try FAISS
    print("6️⃣ Testing FAISS...")
    try:
        import faiss
        print("   ✅ FAISS available")
    except Exception as e:
        print(f"   ❌ FAISS failed: {e}")
        return False
    
    print("\n🎉 ALL CORE COMPONENTS WORK!")
    print("✅ Your RAG system should function properly")
    return True

if __name__ == "__main__":
    if simple_test():
        print("\n🚀 Ready to run: python RAG_app.py")
    else:
        print("\n🔧 Some packages need installation")
        print("Try: python install_packages.py")
