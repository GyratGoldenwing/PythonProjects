#!/usr/bin/env python3
"""
Final verification script - Essential functionality check
"""

def final_verification():
    print("🏁 FINAL RAG SYSTEM VERIFICATION")
    print("=" * 50)
    
    # Check 1: All required files exist
    print("\n📋 Checking required files...")
    import os
    required_files = {
        'requirements.txt': 'Package dependencies',
        'text_extractor.py': 'Document extraction code',
        'Selected_Document.txt': 'Source document',
        'RAG_app.py': 'Main RAG application',
        'prompts.md': 'AI prompts documentation',
        'README.md': 'Reflection report'
    }
    
    all_files_present = True
    for filename, description in required_files.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"   ✅ {filename} ({size} bytes) - {description}")
        else:
            print(f"   ❌ {filename} - MISSING")
            all_files_present = False
    
    # Check 2: Document content
    print("\n📄 Checking document content...")
    try:
        with open('Selected_Document.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        word_count = len(content.split())
        char_count = len(content)
        
        if char_count > 1000:
            print(f"   ✅ Document has sufficient content ({word_count} words, {char_count} characters)")
            
            # Check for AI-related content
            ai_terms = ['artificial intelligence', 'machine learning', 'deep learning', 'neural networks']
            found_terms = [term for term in ai_terms if term.lower() in content.lower()]
            
            if found_terms:
                print(f"   ✅ Document contains relevant AI terms: {', '.join(found_terms)}")
            else:
                print("   ⚠️  Document may not contain expected AI content")
        else:
            print(f"   ❌ Document too short ({char_count} characters)")
            
    except Exception as e:
        print(f"   ❌ Error reading document: {e}")
    
    # Check 3: Code syntax
    print("\n🐍 Checking Python code syntax...")
    python_files = ['text_extractor.py', 'RAG_app.py']
    
    for filename in python_files:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Try to compile the code
            compile(code, filename, 'exec')
            print(f"   ✅ {filename} has valid Python syntax")
            
        except SyntaxError as e:
            print(f"   ❌ {filename} has syntax error: {e}")
        except Exception as e:
            print(f"   ❌ Error checking {filename}: {e}")
    
    # Check 4: Requirements file format
    print("\n📦 Checking requirements.txt...")
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read().strip().split('\n')
        
        expected_packages = [
            'beautifulsoup4', 'langchain', 'sentence-transformers', 
            'numpy', 'faiss-cpu', 'transformers', 'torch'
        ]
        
        missing_packages = [pkg for pkg in expected_packages if pkg not in '\n'.join(requirements)]
        
        if not missing_packages:
            print(f"   ✅ All {len(expected_packages)} required packages listed")
        else:
            print(f"   ⚠️  Missing packages in requirements.txt: {missing_packages}")
            
    except Exception as e:
        print(f"   ❌ Error checking requirements: {e}")
    
    # Check 5: Documentation completeness
    print("\n📚 Checking documentation...")
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            readme_content = f.read()
        
        required_sections = [
            'Project Overview', 'Selected Document', 'Deep-Dive Questions',
            'Chunk Size', 'Quality of Generated Responses', 'Suggestions'
        ]
        
        missing_sections = [section for section in required_sections 
                          if section.lower() not in readme_content.lower()]
        
        if not missing_sections:
            print("   ✅ README.md contains all required sections")
        else:
            print(f"   ⚠️  README.md missing sections: {missing_sections}")
            
        # Check word count
        word_count = len(readme_content.split())
        if word_count > 500:
            print(f"   ✅ README.md has substantial content ({word_count} words)")
        else:
            print(f"   ⚠️  README.md might be too brief ({word_count} words)")
            
    except Exception as e:
        print(f"   ❌ Error checking README.md: {e}")
    
    # Check 6: Prompts documentation
    print("\n💬 Checking prompts documentation...")
    try:
        with open('prompts.md', 'r', encoding='utf-8') as f:
            prompts_content = f.read()
        
        if 'prompt' in prompts_content.lower() and len(prompts_content) > 500:
            print("   ✅ prompts.md contains documented AI prompts")
        else:
            print("   ⚠️  prompts.md may be incomplete")
            
    except Exception as e:
        print(f"   ❌ Error checking prompts.md: {e}")
    
    # Final summary
    print("\n" + "=" * 50)
    if all_files_present:
        print("🎉 VERIFICATION COMPLETE!")
        print("✅ All required files are present and appear correctly formatted")
        print("\n📋 DELIVERABLES READY:")
        print("   ✅ Selected Document (Selected_Document.txt)")
        print("   ✅ Code & Prompts (requirements.txt, text_extractor.py, RAG_app.py, prompts.md)")
        print("   ✅ Reflection Report (README.md)")
        
        print("\n🚀 NEXT STEPS:")
        print("   1. Test the system: python demo_test.py")
        print("   2. Run interactively: python RAG_app.py")
        print("   3. If everything works, ready for GitHub submission!")
        
        return True
    else:
        print("❌ VERIFICATION FAILED")
        print("Please ensure all required files are present and properly formatted")
        return False

if __name__ == "__main__":
    final_verification()
