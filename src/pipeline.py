import os
import qdrant_client
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.schema import ImageNode
from src.utils import convert_pdf_to_images, plot_images
from src.table_extractor import TableExtractor

class PDFTablePipeline:
    def __init__(self, openai_api_key=None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")

        self.table_extractor = TableExtractor()
        self.index = None
        self.retriever_engine = None

    def process_pdf(self, pdf_path, output_dir="pdf_images", qdrant_path="qdrant_index"):
        # 1. Convert PDF to images
        print(f"Converting PDF {pdf_path} to images...")
        image_paths = convert_pdf_to_images(pdf_path, output_dir)

        # 2. Index images
        print("Indexing images...")
        documents_images = SimpleDirectoryReader(output_dir).load_data()

        client = qdrant_client.QdrantClient(path=qdrant_path)
        text_store = QdrantVectorStore(
            client=client, collection_name="text_collection"
        )
        image_store = QdrantVectorStore(
            client=client, collection_name="image_collection"
        )
        storage_context = StorageContext.from_defaults(
            vector_store=text_store, image_store=image_store
        )

        self.index = MultiModalVectorStoreIndex.from_documents(
            documents_images,
            storage_context=storage_context,
        )
        self.retriever_engine = self.index.as_retriever(image_similarity_top_k=2)

        return image_paths

    def query(self, query_text, table_output_dir="table_images"):
        if not self.retriever_engine:
            raise RuntimeError("Pipeline not initialized. Call process_pdf first.")

        print(f"Querying: {query_text}")
        retrieval_results = self.retriever_engine.text_to_image_retrieve(query_text)

        retrieved_image_paths = []
        for res_node in retrieval_results:
            if isinstance(res_node.node, ImageNode):
                retrieved_image_paths.append(res_node.node.metadata["file_path"])

        print(f"Retrieved {len(retrieved_image_paths)} images.")

        # Crop tables from retrieved images
        cropped_table_paths = []
        for file_path in retrieved_image_paths:
            paths = self.table_extractor.detect_and_crop_save_table(
                file_path,
                cropped_table_directory=table_output_dir
            )
            cropped_table_paths.extend(paths)

        if not cropped_table_paths:
            print("No tables detected in retrieved images.")
            # Fallback to using the retrieved images directly?
            # The original notebook seems to assume tables are always found or just proceeds.
            # If we want to answer based on the retrieved page, we can use retrieved_image_paths.
            # But the prompt specifically mentions cropping tables.

        # Generate response
        print("Generating response with GPT-4V...")
        # We need to reload the cropped images as documents for the LLM
        # Or just pass the paths if the API supports it.
        # OpenAIMultiModal.complete takes image_documents which are ImageNode or similar.

        # If cropped_table_paths is empty, SimpleDirectoryReader might fail if directory is empty or not passed correctly.
        # But let's assume we want to pass the cropped tables.

        if cropped_table_paths:
             # Load the cropped tables
             # Note: SimpleDirectoryReader loads all files in the dir.
             # We might want to be specific.
             reader = SimpleDirectoryReader(input_files=cropped_table_paths)
             image_documents = reader.load_data()
        else:
             # Fallback to original retrieved images
             reader = SimpleDirectoryReader(input_files=retrieved_image_paths)
             image_documents = reader.load_data()

        openai_mm_llm = OpenAIMultiModal(
            model="gpt-4-vision-preview",
            api_key=self.openai_api_key,
            max_new_tokens=1500
        )

        response = openai_mm_llm.complete(
            prompt=query_text,
            image_documents=image_documents,
        )

        return response

if __name__ == "__main__":
    # Example usage
    try:
        pipeline = PDFTablePipeline()
        if os.path.exists("llama2.pdf"):
            pipeline.process_pdf("llama2.pdf")
            response = pipeline.query("Compare llama2 with llama1?")
            print(response)
        else:
            print("Example PDF 'llama2.pdf' not found. Please provide a valid PDF file.")
    except Exception as e:
        print(f"Error: {e}")
