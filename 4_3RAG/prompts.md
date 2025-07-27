# AI Prompts I Used for This Project

I used ChatGPT and Claude to help me build this RAG system. Here are the actual prompts I used (some worked better than others):

## Getting Started - Requirements File

**Prompt:** "I need to build a RAG system for a class project. What Python packages do I need? Give me a requirements.txt file for: beautifulsoup4, langchain, sentence-transformers, numpy, faiss-cpu, transformers, torch"

This worked pretty well, though I had to add version numbers later when stuff wasn't working.

## Text Extraction Code

**First attempt (didn't work great):**
"Write Python code to scrape Wikipedia and save it as a text file"

**Better prompt that worked:**
"Write a Python function called scrape_webpage(url) that uses requests to fetch a Wikipedia page, parses it with BeautifulSoup, extracts all the paragraph text from the main content area, and saves it to Selected_Document.txt. Make sure to handle errors and use UTF-8 encoding. I want to scrape the Wikipedia page about artificial intelligence."

## RAG System Components

### Document Reading
**Prompt:** "Write Python code to read a text file called Selected_Document.txt into a variable called text using UTF-8 encoding"

Simple but I needed to make sure I didn't mess up the encoding.

### Text Chunking  
**Prompt:** "Show me how to use LangChain's RecursiveCharacterTextSplitter to split text into 500-character chunks with 50 characters of overlap. Use separators for paragraphs, sentences, and words."

Had to ask a follow-up about what separators to use exactly.

### Embeddings and FAISS
**Prompt:** "Write code to load the sentence-transformers model 'all-distilroberta-v1', encode a list of text chunks, convert to numpy float32, create a FAISS IndexFlatL2, and add the embeddings to it"

This one took a few tries because I didn't understand the numpy conversion at first.

### Text Generation
**Prompt:** "How do I set up a HuggingFace text2text-generation pipeline using the google/flan-t5-small model to run on CPU?"

**Follow-up prompt:** "The model is taking forever to load, is there a smaller/faster option?"
(Turns out flan-t5-small is already pretty small)

### Retrieval Function
**Prompt:** "Write a function that takes a question, encodes it with sentence transformers, searches a FAISS index, and returns the top 5 most similar text chunks"

### Answer Generation  
**Prompt:** "Write a function that takes a question, retrieves relevant chunks, builds a prompt with context, and generates an answer using the FLAN-T5 model. Format it nicely with clear Context and Question sections."

## Error Handling and Improvements

**When stuff was breaking:**
"My RAG system crashes when packages aren't installed. How do I add better error handling and informative error messages?"

**For the interactive loop:**
"Write a while loop that keeps asking the user for questions and generates answers until they type 'exit' or 'quit'. Add some example questions to help users get started."

## Deep Dive Questions for Understanding

**Main prompt:** "I built a RAG system for a class project. Generate 5 technical questions that would help me understand how it works, covering topics like embeddings, vector search, chunking strategy, and prompt engineering. Then provide detailed answers that a student could understand."

This was super helpful for writing the reflection section.

## Debugging Prompts (The Ones I Needed When Things Broke)

**Package issues:**
"I'm getting 'No module named transformers.logging' error. How do I fix this?"

**Model loading problems:**
"My sentence transformer model won't load, it says something about CUDA. I'm on a Mac, help!"

**FAISS errors:**
"FAISS is giving me dimension mismatch errors, what am I doing wrong?"

**Performance issues:**
"My RAG system is super slow, how do I speed it up without breaking it?"

## Experiment Design

**For chunk size testing:**
"Design an experiment to test different chunk sizes (250, 500, 1000 characters) and overlap amounts (25, 50, 100, 150) for a RAG system. What metrics should I look at and how do I structure the comparison?"

**For quality evaluation:**
"How do I evaluate if my RAG system is giving good answers? What should I test it with?"

## Mistakes I Made

1. **First prompt was too vague:** "Build me a RAG system" → Got generic code that didn't work
2. **Forgot to specify versions:** Led to package compatibility issues
3. **Didn't ask about error handling:** Had to go back and add try/catch blocks everywhere  
4. **Assumed models would be fast:** Didn't realize about download times and model sizes

## What Worked Best

- **Being specific** about exactly what I wanted the code to do
- **Including example inputs/outputs** when I could
- **Asking for error handling** upfront instead of adding it later
- **Breaking complex tasks into smaller prompts** instead of trying to get everything at once

## Prompts I Wish I'd Used Earlier

"What are common gotchas when building RAG systems that beginners run into?"

"Explain the trade-offs between different embedding models for RAG applications"

"How do I know if my chunk size is appropriate for my document type?"

These would have saved me a lot of trial and error!
