#!/usr/bin/env python3
"""
Improved RAG system test with better error handling
"""

def test_package_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing Package Imports...")
    
    # Test basic packages first
    try:
        import logging
        import warnings
        import numpy as np
        print("   ✅ Basic packages (logging, warnings, numpy) imported")
    except ImportError as e:
        print(f"   ❌ Basic packages failed: {e}")
        return False
    
    # Test transformers
    try:
        import transformers
        print(f"   ✅ Transformers {transformers.__version__} imported")
    except ImportError as e:
        print(f"   ❌ Transformers import failed: {e}")
        print("   💡 Try: pip install transformers")
        return False
    
    # Test transformers logging (optional)
    try:
        import transformers.logging as hf_logging
        print("   ✅ Transformers logging imported")
    except ImportError:
        print("   ⚠️  Transformers logging not available (using fallback)")
    
    # Test other ML packages
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__} imported")
    except ImportError as e:
        print(f"   ❌ PyTorch import failed: {e}")
        print("   💡 Try: pip install torch")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("   ✅ Sentence Transformers imported")
    except ImportError as e:
        print(f"   ❌ Sentence Transformers import failed: {e}")
        print("   💡 Try: pip install sentence-transformers")
        return False
    
    try:
        import faiss
        print("   ✅ FAISS imported")
    except ImportError as e:
        print(f"   ❌ FAISS import failed: {e}")
        print("   💡 Try: pip install faiss-cpu")
        return False
    
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        print("   ✅ LangChain imported")
    except ImportError as e:
        print(f"   ❌ LangChain import failed: {e}")
        print("   💡 Try: pip install langchain")
        return False
    
    return True

def test_document_loading():
    """Test document file loading"""
    print("🔍 Testing Document Loading...")
    try:
        with open('Selected_Document.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if len(content) > 100:
            print(f"   ✅ Document loaded successfully ({len(content)} characters)")
            return True
        else:
            print("   ❌ Document file is empty or too short")
            return False
    except FileNotFoundError:
        print("   ❌ Selected_Document.txt not found")
        return False
    except Exception as e:
        print(f"   ❌ Error loading document: {e}")
        return False

def test_text_chunking():
    """Test text splitting functionality"""
    print("🔍 Testing Text Chunking...")
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        with open('Selected_Document.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', ' ', ''],
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_text(text)
        
        if len(chunks) > 0:
            print(f"   ✅ Text splitting successful ({len(chunks)} chunks created)")
            return True
        else:
            print("   ❌ No chunks created")
            return False
    except Exception as e:
        print(f"   ❌ Text chunking failed: {e}")
        return False

def test_quick_model_load():
    """Test loading a simple model"""
    print("🔍 Testing Quick Model Load...")
    try:
        from transformers import pipeline
        
        # Try to load a small model for testing
        print("   ⏳ Loading small test model...")
        generator = pipeline('text2text-generation', model='google/flan-t5-small', device=-1)
        
        # Quick test
        result = generator("Question: What is AI? Answer:", max_length=20)
        print(f"   ✅ Model loaded and tested successfully")
        print(f"   🤖 Test output: {result[0]['generated_text']}")
        return True
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        print("   💡 This might be due to network issues or missing packages")
        return False

def main():
    print("🧪 IMPROVED RAG SYSTEM TEST")
    print("=" * 40)
    
    tests = [
        ("Package Imports", test_package_imports),
        ("Document Loading", test_document_loading), 
        ("Text Chunking", test_text_chunking),
        ("Quick Model Load", test_quick_model_load)
    ]
    
    passed = 0
    failed_tests = []
    
    for test_name, test_func in tests:
        print()
        try:
            if test_func():
                passed += 1
            else:
                failed_tests.append(test_name)
        except Exception as e:
            print(f"   ❌ {test_name} crashed: {e}")
            failed_tests.append(test_name)
    
    print(f"\n{'='*40}")
    print(f"📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! RAG system is ready.")
        print("\n🚀 Next steps:")
        print("   • Run: python RAG_app.py")
        print("   • Ask questions about AI")
    elif passed >= len(tests) - 1:
        print("✅ Core functionality working!")
        print("⚠️  One test failed, but system may still work")
        print("\n🚀 Try running: python RAG_app.py")
    else:
        print(f"❌ {len(failed_tests)} test(s) failed:")
        for test in failed_tests:
            print(f"   • {test}")
        print("\n🔧 Install missing packages:")
        print("   pip install transformers torch sentence-transformers faiss-cpu langchain")

if __name__ == "__main__":
    main()
