## Overview
This project is a simple POC to explore code summarization using a local LLM (Ollama).

It reads code files from a repository, splits them into smaller chunks, and generates summaries to help understand large codebases.

## Tech Stack
- Ollama (Local LLM inference)  
- Python  
- LangChain (orchestration)  

## Setup Instructions

### 1. Install Ollama
Download and install Ollama:
https://ollama.com/download  

Pull the model:

ollama pull llama3


### 2. Install Dependencies

py -m pip install -r requirements.txt

## How to Run

### Step 1: Process Code Files
This step reads code from the repository, splits it into chunks, and prepares it for summarization.

py src/ingest.py

### Step 2: Generate Summaries
This step generates summaries and basic documentation from the processed code.

py src/query.py


## Features
- Read code files from a repository  
- Split large files into smaller chunks  
- Generate summaries using LLM  
- Basic documentation generation  

## Flow
1. Load code files  
2. Split into chunks  
3. Send chunks to LLM  
4. Generate summaries  
5. Combine summaries for output  

## Outcome
This POC helps in understanding how LLMs can be used for code summarization and documentation, especially for large codebases.

## Use Cases
- Code understanding  
- Developer onboarding  
- Documentation generation  
- Legacy system analysis  
