#!/usr/bin/env python3
"""
Quick functional test for RAG system core components
"""

def quick_test():
    print("🚀 Quick RAG System Test")
    print("=" * 30)
    
    try:
        # Test 1: Document loading
        print("1️⃣ Testing document loading...")
        with open('Selected_Document.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"   ✅ Loaded {len(text)} characters")
        
        # Test 2: Text splitting
        print("2️⃣ Testing text splitting...")
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)
        print(f"   ✅ Created {len(chunks)} chunks")
        
        # Test 3: Check if we have the document content we expect
        print("3️⃣ Testing document content...")
        if "artificial intelligence" in text.lower():
            print("   ✅ Document contains AI content")
        else:
            print("   ⚠️  Document might not have expected content")
        
        # Test 4: Basic imports
        print("4️⃣ Testing critical imports...")
        import numpy as np
        import torch
        print("   ✅ NumPy and PyTorch imported")
        
        # Test 5: File structure
        print("5️⃣ Testing file structure...")
        import os
        required_files = ['RAG_app.py', 'requirements.txt', 'README.md', 'prompts.md']
        missing = [f for f in required_files if not os.path.exists(f)]
        if not missing:
            print("   ✅ All required files present")
        else:
            print(f"   ❌ Missing files: {missing}")
        
        print("\n🎯 Quick test completed successfully!")
        print("   Ready to run full verification or RAG application")
        
    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        print("   Check package installation and file contents")

if __name__ == "__main__":
    quick_test()
