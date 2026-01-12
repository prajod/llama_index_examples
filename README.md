# PDF Table Extractor

This repository demonstrates how to extract tables from PDF files using Microsoft's Table Transformer and answer questions about them using GPT-4V, leveraging LlamaIndex and Qdrant.

## Installation

1.  Clone the repository.
2.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3.  Set up your OpenAI API key in a `.env` file or export it as an environment variable:

    ```bash
    export OPENAI_API_KEY="your-api-key"
    ```

## Usage

### Using the Python Module

You can use the `PDFTablePipeline` class in `src/pipeline.py` to run the extraction and querying process.

```python
from src.pipeline import PDFTablePipeline

pipeline = PDFTablePipeline()

# Process the PDF (convert to images, index, etc.)
# This will save images to 'pdf_images/' and tables to 'table_images/'
pipeline.process_pdf("path/to/your.pdf")

# Query the index
response = pipeline.query("Compare llama2 with llama1?")
print(response)
```

### Running the Notebook

The original notebook `table_transformer_gpt4v_pdf_tables.ipynb` is also available but has been refactored into the `src` directory for better maintainability and testing.

## Code Structure

*   `src/table_extractor.py`: Contains the `TableExtractor` class which uses Microsoft's Table Transformer to detect and crop tables.
*   `src/utils.py`: Helper functions for PDF conversion and image plotting.
*   `src/pipeline.py`: The main pipeline class that orchestrates the PDF processing, indexing, and querying using LlamaIndex.

## Testing

Run the tests using `unittest`:

```bash
python3 -m unittest discover tests
```

## Improvements over the original example

*   **Refactoring**: Logic extracted into reusable modules (`src/`).
*   **Robustness**: Better path handling (using `os.path.join`, `os.path.exists`), error handling, and `try-except` blocks.
*   **Dependencies**: Updated `requirements.txt` and compatibility with newer `llama-index` versions.
*   **Testing**: Added unit tests for the table extractor logic.
*   **Bug Fixes**: Fixed path issues where directories might be incorrectly concatenated.
