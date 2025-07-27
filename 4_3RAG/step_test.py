#!/usr/bin/env python3
"""
Step-by-step RAG system test
"""

def test_step_by_step():
    print("🔧 STEP-BY-STEP RAG SYSTEM VERIFICATION")
    print("=" * 50)
    
    # Step 1: Basic imports
    print("\n1️⃣ Testing basic imports...")
    try:
        import logging
        import warnings
        import os
        print("   ✅ Basic Python modules imported")
    except Exception as e:
        print(f"   ❌ Basic imports failed: {e}")
        return False
    
    # Step 2: NumPy and basic ML
    print("\n2️⃣ Testing NumPy...")
    try:
        import numpy as np
        test_array = np.array([1, 2, 3])
        print(f"   ✅ NumPy working (test array: {test_array})")
    except Exception as e:
        print(f"   ❌ NumPy failed: {e}")
        return False
    
    # Step 3: PyTorch
    print("\n3️⃣ Testing PyTorch...")
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__} imported")
    except Exception as e:
        print(f"   ❌ PyTorch failed: {e}")
        return False
    
    # Step 4: Transformers
    print("\n4️⃣ Testing Transformers...")
    try:
        import transformers
        print(f"   ✅ Transformers {transformers.__version__} imported")
    except Exception as e:
        print(f"   ❌ Transformers failed: {e}")
        return False
    
    # Step 5: Sentence Transformers
    print("\n5️⃣ Testing Sentence Transformers...")
    try:
        from sentence_transformers import SentenceTransformer
        print("   ✅ Sentence Transformers imported")
    except Exception as e:
        print(f"   ❌ Sentence Transformers failed: {e}")
        return False
    
    # Step 6: FAISS
    print("\n6️⃣ Testing FAISS...")
    try:
        import faiss
        print(f"   ✅ FAISS imported")
    except Exception as e:
        print(f"   ❌ FAISS failed: {e}")
        return False
    
    # Step 7: LangChain
    print("\n7️⃣ Testing LangChain...")
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        print("   ✅ LangChain text splitter imported")
    except Exception as e:
        print(f"   ❌ LangChain failed: {e}")
        return False
    
    # Step 8: Document loading
    print("\n8️⃣ Testing document loading...")
    try:
        with open('Selected_Document.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"   ✅ Document loaded ({len(text)} characters)")
        if len(text) < 100:
            print("   ⚠️  Document seems too short")
    except Exception as e:
        print(f"   ❌ Document loading failed: {e}")
        return False
    
    # Step 9: Text splitting
    print("\n9️⃣ Testing text splitting...")
    try:
        splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', ' ', ''],
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_text(text)
        print(f"   ✅ Text split into {len(chunks)} chunks")
        if len(chunks) > 0:
            print(f"   📝 First chunk preview: {chunks[0][:100]}...")
    except Exception as e:
        print(f"   ❌ Text splitting failed: {e}")
        return False
    
    # Step 10: Model loading (this might take time)
    print("\n🔟 Testing model loading (this may take a while)...")
    try:
        # Use a very simple model for testing
        from transformers import pipeline
        print("   ⏳ Loading FLAN-T5-small...")
        generator = pipeline('text2text-generation', model='google/flan-t5-small', device=-1)
        
        # Quick test
        test_result = generator("Question: What is 2+2? Answer:", max_length=20)
        print(f"   ✅ Generation model loaded and tested")
        print(f"   🤖 Test output: {test_result[0]['generated_text']}")
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        print("   💡 This might be due to network issues or insufficient disk space")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
    print("Your RAG system should be fully functional.")
    print("\n🚀 Next steps:")
    print("   • Run: python RAG_app.py")
    print("   • Ask questions about artificial intelligence")
    print("   • Test with queries like 'What is machine learning?'")
    
    return True

if __name__ == "__main__":
    success = test_step_by_step()
    if not success:
        print("\n🔧 Some components need attention before the RAG system will work properly.")
