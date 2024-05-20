import os
import configparser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorstoreIndexCreator, VectorStoreIndexWrapper
from langchain_community.llms import Bedrock

def get_dynamic_params():
    config = configparser.ConfigParser()
    # Construct the path to the PDF file dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parameters_file_path = os.path.join(current_dir, '..', 'parameters.ini')
    config.read(parameters_file_path)
    
    vector_index_params = config['VectorIndexParams']

    params = {}
    params['file_name'] = vector_index_params.get('file_name', '')
    params['text_splitter'] = {
        'chunk_size': int(vector_index_params.get('chunk_size', '100')),
        'chunk_overlap': int(vector_index_params.get('chunk_overlap', '10'))
    }
    params['aws_cred_profile'] = vector_index_params.get('aws_cred_profile', 'default')
    params['embedding_model_id'] = vector_index_params.get('embedding_model_id', 'amazon.titan-embed-text-v1')
    params['folder'] = vector_index_params.get("folder_name")

    return params
    
def get_llm_parameters():
    config = configparser.ConfigParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parameters_file_path = os.path.join(current_dir, '..', 'parameters.ini')
    config.read(parameters_file_path)
    
    llm_parameters = config['LLMParameters']

    params = {}
    params['cred_profile'] = llm_parameters.get('cred_profile')
    params['model_id'] = llm_parameters.get('model_id')
    params['model_params'] = {
        'max_gen_len': int(llm_parameters.get('max_gen_len')),
        'temperature': float(llm_parameters.get('temperature')),
        'top_p': float(llm_parameters.get('top_p'))
    }

    return params

def load_pdfs_from_folder(folder_path):
    print(f"Reading the pdfs in the location {folder_path}")
    data_loaded_ist = []
    
    for filename in os.listdir(folder_path):     
        if filename.endswith('.pdf'):
            print(f"Reading data for {filename}")
            # Construct the full file path
            pdf_path = os.path.join(folder_path, filename)
            # Loading the pdf data in PyPDFLoader Object
            pdf_loader_data = PyPDFLoader(pdf_path)
            data_loaded_ist.append(pdf_loader_data)
                   
    return data_loaded_ist
    

def get_document_index():
    # Get Dynamic Parameters
    params = get_dynamic_params()
    
    # Loading the pdf data in PyPDFLoader Object
    # Construct the path to the PDF file dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))   
    folder_location = os.path.join(current_dir, '..', params["folder"])
    pdf_loader_data_list = load_pdfs_from_folder(folder_location)
    print(pdf_loader_data_list)

    # Creating Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", "" ],
        chunk_size = params["text_splitter"]["chunk_size"],
        chunk_overlap = params["text_splitter"]["chunk_overlap"]
    )

    # Instantiating BedrockEmbedding
    bedrock_embedding = BedrockEmbeddings(
        credentials_profile_name=params["aws_cred_profile"],
        model_id=params["embedding_model_id"]
    )

    # Splitting the Db, Creating VectorDB, Creting Embeddings and Storing it using VectorStoreIndexCreator
    index_creator = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        text_splitter=text_splitter,
        embedding=bedrock_embedding
    )

    db_index = index_creator.from_loaders(pdf_loader_data_list)
    return db_index

def fm_llm():
    params = get_llm_parameters()
       
    # Create an LLM for Bedrock Model
    return Bedrock(
        credentials_profile_name=params["cred_profile"],
        model_id=params["model_id"],
        model_kwargs=params["model_params"]
        )
    
def doc_rag_response(index: VectorStoreIndexWrapper, prompt):
    # Query from the index
    hr_rag_query  = index.query(
        question=prompt,
        llm=fm_llm()
    )
    return hr_rag_query