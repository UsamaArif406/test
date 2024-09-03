import streamlit as st
import nest_asyncio
import os
import re
from llama_index.core.schema import TextNode
from typing import Optional
from pathlib import Path
from llama_index.core import Settings, StorageContext, SummaryIndex, load_index_from_storage, set_global_handler
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_parse import LlamaParse
from pydantic.v1 import BaseModel, Field
from typing import List
from IPython.display import display, Markdown, Image

# # Setup Arize Phoenix for logging/observability
# PHOENIX_API_KEY = "<PHOENIX_API_KEY>"
# os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
# set_global_handler("arize_phoenix", endpoint="https://llamatrace.com/v1/traces")

# Apply nest_asyncio
nest_asyncio.apply()

# Create directories for data and images if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("data_images", exist_ok=True)

# Streamlit app
st.title('Multimodal Report Generation')

uploaded_file = st.file_uploader('Upload a PDF file', type='pdf')

if uploaded_file is not None:
    # Save the uploaded file
    file_path = f'data/{uploaded_file.name}'
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    st.success('File uploaded successfully!')

    # Initialize LlamaParse and OpenAI
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    llm = OpenAI(model="gpt-4o")
    Settings.embed_model = embed_model
    Settings.llm = llm

    parser = LlamaParse(
        result_type="markdown",
        use_vendor_multimodal_model=True,
        vendor_multimodal_model_name="anthropic-sonnet-3.5",
    )

    st.write("Parsing slide deck...")
    md_json_objs = parser.get_json_result(file_path)
    md_json_list = md_json_objs[0]["pages"]

    # Extract and display images
    image_dicts = parser.get_images(md_json_objs, download_path="data_images")

    def get_page_number(file_name):
        match = re.search(r"-page-(\d+)\.jpg$", str(file_name))
        if match:
            return int(match.group(1))
        return 0

    def _get_sorted_image_files(image_dir):
        """Get image files sorted by page."""
        raw_files = [f for f in list(Path(image_dir).iterdir()) if f.is_file()]
        sorted_files = sorted(raw_files, key=get_page_number)
        return sorted_files

    def get_text_nodes(json_dicts, image_dir=None):
        """Split docs into nodes, by separator."""
        nodes = []

        image_files = _get_sorted_image_files(image_dir) if image_dir is not None else None
        md_texts = [d["md"] for d in json_dicts]

        for idx, md_text in enumerate(md_texts):
            chunk_metadata = {"page_num": idx + 1}
            if image_files is not None:
                image_file = image_files[idx]
                chunk_metadata["image_path"] = str(image_file)
            chunk_metadata["parsed_text_markdown"] = md_text
            node = TextNode(
                text="",
                metadata=chunk_metadata,
            )
            nodes.append(node)

        return nodes

    # Get text nodes
    text_nodes = get_text_nodes(md_json_list, image_dir="data_images")
    st.write(f"Processed {len(text_nodes)} pages.")

    # Save or load the summary index
    if not os.path.exists("storage_nodes_summary"):
        index = SummaryIndex(text_nodes)
        index.set_index_id("summary_index")
        index.storage_context.persist("./storage_nodes_summary")
        st.success("Summary index created and stored successfully!")
    else:
        storage_context = StorageContext.from_defaults(persist_dir="storage_nodes_summary")
        index = load_index_from_storage(storage_context, index_id="summary_index")
        st.success("Summary index loaded from storage!")

    # Data model for report
    class TextBlock(BaseModel):
        """Text block."""
        text: str = Field(..., description="The text for this block.")

    class ImageBlock(BaseModel):
        """Image block."""
        file_path: str = Field(..., description="File path to the image.")

    class ReportOutput(BaseModel):
        """Data model for a report."""
        blocks: List[TextBlock | ImageBlock] = Field(
            ..., description="A list of text and image blocks."
        )

        def render(self) -> None:
            """Render as HTML on the page."""
            for b in self.blocks:
                if isinstance(b, TextBlock):
                    st.markdown(b.text)
                else:
                    st.image(b.file_path)

    # LLM with structured output
    system_prompt = """\
    You are a report generation assistant tasked with producing a well-formatted context given parsed context.

    You will be given context from one or more reports that take the form of parsed text.

    You are responsible for producing a report with interleaving text and images - in the format of interleaving text and "image" blocks.
    Since you cannot directly produce an image, the image block takes in a file path - you should write in the file path of the image instead.

    How do you know which image to generate? Each context chunk will contain metadata including an image render of the source chunk, given as a file path. 
    Include ONLY the images from the chunks that have heavy visual elements (you can get a hint of this if the parsed text contains a lot of tables).
    You MUST include at least one image block in the output.

    You MUST output your response as a tool call in order to adhere to the required output format. Do NOT give back normal text.
    """

    llm = OpenAI(model="gpt-4o", system_prompt=system_prompt)
    sllm = llm.as_structured_llm(output_cls=ReportOutput)
    query_engine = index.as_query_engine(
        similarity_top_k=10,
        llm=sllm,
        response_mode="compact",
    )

    # Query the engine and render the report
    query = st.text_input("Enter your query", "Give me a summary of the financial performance of the Alaska/International segment vs. the lower 48 segment")
    if query:
        response = query_engine.query(query)
        response.response.render()

else:
    st.info("Please upload a PDF file to start the process.")
