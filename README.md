# GEMINI Assistant CLI

A lightweight, elegant command-line interface for the Gemini AI model. Built with Python and OpenAI compatibility layer.

## Overview

This project provides a simple yet powerful CLI assistant powered by the `gemini-3.8-flash` model. It leverages the OpenAI client libraries configured to point to Google's Generative AI API endpoints, combining the best of both ecosystems.

## Features
- **Seamless Chat Interface:** Interactive command-line chat application.
- **Modern Tooling:** Uses `uv` for lightning-fast dependency management (see `pyproject.toml`).
- **OpenAI Compatibility:** Built using the `openai` Python SDK targeting Gemini.

## Requirements

- Python >= 3.14
- Dependencies dependencies managed by `uv` or `pip`:
  - `mcp>=2.3.0`
  - `openai>=3.26.0`
  - `pydantic>=2.13.5`

## Setup & Usage

1. Create a `.env` file or export your Gemini API key:
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```

2. Run the assistant:
   ```bash
   python v0.py
   ```

3. Press `Ctrl+C` to quit the interactive session.
