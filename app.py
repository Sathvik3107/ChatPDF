import gradio as gr
import time
import os
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from pinecone import Pinecone
from openai import OpenAI

# Initialize OpenAI and Pinecone clients
op_key = "OpenAIKey"
pc_key = "PineconeKey"
os.environ['OPENAI_API_KEY'] = op_key
client = OpenAI(api_key=op_key)
pc = Pinecone(api_key=pc_key)

def read_documents(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents

def split_documents(documents, chunk_size=1300, chunk_overlap=40):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

def index_documents(documents, index_name):
    index = pc.Index(index_name)
    for i, doc in enumerate(documents):
        page_id = f"page_{i}"
        res = client.embeddings.create(input=doc.page_content, model="text-embedding-3-small")
        embedding_vector = res.data[0].embedding
        index.upsert(vectors=[{"id": page_id, "values": embedding_vector}])

def retrieve_contexts(index, query, all_text, limit=3750):
    res = client.embeddings.create(input=[query], model="text-embedding-3-small")
    xq = res.data[0].embedding

    contexts = []
    time_waited = 0
    while (len(contexts) < 3 and time_waited < 60 * 12):
        res = index.query(vector=xq, top_k=3, include_metadata=True)
        contexts = contexts + [
            all_text[int(x['id'].split('_')[1])] for x in res['matches']
        ]
        print(f"Retrieved {len(contexts)} contexts, sleeping for 15 seconds...")
        time.sleep(15)
        time_waited += 15

    if time_waited >= 60 * 12:
        print("Timed out waiting for contexts to be retrieved.")
        contexts = ["No contexts retrieved. Try to answer the question yourself!"]

    prompt_start = "Answer the question based on the context below.\n\n" + "Context:\n"
    prompt_end = f"\n\nQuestion: {query}\nAnswer:"
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = prompt_start + "\n\n---\n\n".join(contexts[:i-1]) + prompt_end
            break
        elif i == len(contexts)-1:
            prompt = prompt_start + "\n\n---\n\n".join(contexts) + prompt_end
    return prompt

def generate_response(prompt):
    sys_prompt = "You are a helpful assistant that always answers questions."
    res = client.chat.completions.create(
        model='gpt-3.5-turbo-0613',
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return res.choices[0].message.content.strip()

def respond_to_message(message, chat_history):
    query_context = retrieve_contexts(message, all_text)
    bot_message = generate_response(query_context)
    chat_history.append((message, bot_message))
    return "", chat_history

def start_chat():
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        Submit = gr.ClearButton([msg, chatbot])

        msg.submit(respond_to_message, [msg, chatbot], [msg, chatbot])

        demo.launch()

if __name__ == "__main__":
    # Read documents
    documents = read_documents('/Users/sathvikptari/Desktop/prj3')

    # Split documents
    docs = split_documents(documents)

    # Index documents
    index_documents(docs, "myindex")

    # Prepare all_text
    all_text = [doc.page_content for doc in docs]

    # Start chat interface
    start_chat()
