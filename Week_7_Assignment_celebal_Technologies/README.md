# Document Question Answering System (RAG)

## Overview

This project is a simple Document Question Answering System built using LangChain and OpenAI.

It loads custom PDF or Text documents and answers user questions only based on the content of those documents.

## Features

- Load PDF documents
- Load Text documents
- Answer questions from uploaded documents
- Uses LangChain Document Loaders
- Maintains conversation history
- Returns "I don't know" if the answer is not present in the document

## Files

- chatbot_pdf_loader.py
- chatbot_text_loader.py
- environment.pdf
- cricket.txt

## Requirements

- Python 3.10+
- OpenAI API Key

## Installation

Install dependencies

```bash
pip install -r requirements.txt
