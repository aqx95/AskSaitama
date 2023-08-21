import sys
import logging
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, get_response_synthesizer
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# LLM pipe
llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_url="https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q4_0.bin",
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": -1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)


def ask(file, query_):
    logger = logging.getLogger(__name__)

    # Dataloader
    docs = SimpleDirectoryReader(input_files=[file]).load_data()

    # Use huggingface model for embeddings
    service_context = ServiceContext.from_defaults(
        embed_model="local:sentence-transformers/all-MiniLM-L6-v2",
        llm=llm)

    # Build index
    logger.info("Building index from collection of documents...")
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)

    # Retriever pipe
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5)
    reranker = SentenceTransformerRerank(
        model='cross-encoder/ms-marco-MiniLM-L-6-v2',
        top_n=3)
    
    # Response pipe
    response_synth = get_response_synthesizer(
        response_mode='compact', 
        service_context=service_context,
        streaming=True)
    
    # Query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synth,
        node_postprocessors=[reranker])
    
    logger.info("Generating response....")
    response = query_engine.query(query_)
    return response


if __name__ == '__main__':
    query_ = input("Ask me anything! \n")
    ask('/home/ml_bob/work/info_retrieval/jake_code/garden.pdf', query_)
