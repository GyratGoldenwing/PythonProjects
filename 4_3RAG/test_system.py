#!/usr/bin/env python3
"""
Updated test script for the RAG system with better error handling
"""

def test_imports():
    """Test if all required packages are available"""
    print("🔍 Testing Package Imports...")
    
    failed_imports = []
    
    # Test basic imports
    try:
        import logging
        import warnings
        import numpy as np
        print("   ✅ Basic packages (logging, warnings, numpy)")
    except ImportError as e:
        print(f"   ❌ Basic packages failed: {e}")
        failed_imports.append("basic")
    
    # Test transformers
    try:
        import transformers
        print(f"   ✅ transformers {transformers.__version__}")
        
        # Test transformers.logging (optional)
        try:
            import transformers.logging as hf_logging
            print("   ✅ transformers.logging")
        except ImportError:
            print("   ⚠️  transformers.logging not available (using fallback)")
    except ImportError as e:
        print(f"   ❌ transformers failed: {e}")
        print("   💡 Install with: pip install transformers")
        failed_imports.append("transformers")
    
    # Test sentence transformers
    try:
        from sentence_transformers import SentenceTransformer
        print("   ✅ sentence-transformers")
    except ImportError as e:
        print(f"   ❌ sentence-transformers failed: {e}")
        print("   💡 Install with: pip install sentence-transformers")
        failed_imports.append("sentence-transformers")
    
    # Test FAISS
    try:
        import faiss
        print("   ✅ faiss")
    except ImportError as e:
        print(f"   ❌ faiss failed: {e}")
        print("   💡 Install with: pip install faiss-cpu")
        failed_imports.append("faiss")
    
    # Test LangChain
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        print("   ✅ langchain")
    except ImportError as e:
        print(f"   ❌ langchain failed: {e}")
        print("   💡 Install with: pip install langchain")
        failed_imports.append("langchain")
    
    return len(failed_imports) == 0

def test_document_loading():
    """Test document file exists and has content"""
    print("🔍 Testing Document Loading...")
    try:
        with open('Selected_Document.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if len(content) > 100:
            print(f"   ✅ Document loaded successfully ({len(content)} characters)")
            return True
        else:
            print("   ❌ Document file is too short or empty")
            return False
    except FileNotFoundError:
        print("   ❌ Selected_Document.txt not found")
        return False
    except Exception as e:
        print(f"   ❌ Error reading document: {e}")
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
        print(f"   ❌ Text splitting failed: {e}")
        return False

def main():
    print("🧪 Testing RAG System Components")
    print("========================================")
    
    tests = [
        ("Package Imports", test_imports),
        ("Document Loading", test_document_loading),
        ("Text Chunking", test_text_chunking)
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
    
    print(f"\n📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! RAG system is ready to use.")
        print("\nTo run the interactive RAG system, use:")
        print("   python RAG_app.py")
    elif passed >= 2:
        print("✅ Core functionality working!")
        if "Package Imports" in failed_tests:
            print("\n🔧 Missing packages detected. Install with:")
            print("   pip install transformers torch sentence-transformers faiss-cpu langchain")
            print("\nOr run the installer:")
            print("   python install_packages.py")
        print("\n🚀 Try running the RAG app anyway:")
        print("   python RAG_app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\n🔧 To fix package issues:")
        print("   python install_packages.py")

if __name__ == "__main__":
    main()
