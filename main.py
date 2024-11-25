import streamlit as st
import boto3
import os
from botocore.exceptions import NoCredentialsError
from langchain.document_loaders import AmazonTextractPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import json
from typing import Dict

def get_secrets() -> Dict[str, str]:
    required_secrets = [
        'OPENAI_API_KEY',
        'LLAMA_CLOUD_API_KEY',
        'AWS_ACCESS_KEY',
        'AWS_SECRET_KEY',
        'AWS_BUCKET_NAME',
        'AWS_REGION'
    ]
    
    missing_secrets = [secret for secret in required_secrets if secret not in st.secrets]
    if missing_secrets:
        raise ValueError(f"Missing required secrets: {', '.join(missing_secrets)}")
    
    return {secret: st.secrets[secret] for secret in required_secrets}

# Set environment variables from secrets
secrets = get_secrets()
os.environ["OPENAI_API_KEY"] = secrets['OPENAI_API_KEY']
os.environ['Llama_Cloud_API_Key'] = secrets['LLAMA_CLOUD_API_KEY']

s3_client = boto3.client(
    's3',
    aws_access_key_id=secrets['AWS_ACCESS_KEY'],
    aws_secret_access_key=secrets['AWS_SECRET_KEY'],
    region_name=secrets['AWS_REGION']
)

textract_client = boto3.client(
    'textract',
    aws_access_key_id=secrets['AWS_ACCESS_KEY'],
    aws_secret_access_key=secrets['AWS_SECRET_KEY'],
    region_name=secrets['AWS_REGION']
)

def upload_to_s3(file, filename, folder="cards"):
    try:
        s3_path = f"{folder}/{filename}"
        s3_client.upload_fileobj(file, secrets['AWS_BUCKET_NAME'], s3_path)
        return f"s3://{secrets['AWS_BUCKET_NAME']}/{s3_path}"
    except NoCredentialsError:
        return None

def process_and_query(text, query):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32, length_function=len)
    texts = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    docs = docsearch.similarity_search(query)
    return chain.run(input_documents=docs, question=query)

def process_card_side(file_path):
    loader = AmazonTextractPDFLoader(file_path, client=textract_client)
    documents = loader.load()
    return documents[0].page_content

def main():
    st.set_page_config(page_title="Emirates Card Processor", page_icon="🆔", layout="wide")

    st.markdown("""
        <style>
            .stApp { 
                background-color: #1a1a1a;
            }
            .main-title {
                color: white;
                text-align: center;
                font-size: 2rem;
                margin-bottom: 1.5rem;
            }
            .upload-label {
                color: #4a90e2;
                font-size: 1.1rem;
                font-weight: 500;
                margin-bottom: 0.5rem;
            }
            .upload-container {
                background: rgba(74, 144, 226, 0.1);
                border: 1px solid rgba(74, 144, 226, 0.3);
                border-radius: 6px;
                padding: 1rem;
            }
            .stButton > button {
                background: #1e3d59;
                color: white;
                border: none;
                padding: 0.75rem 3rem;
                border-radius: 6px;
                font-weight: 500;
                transition: all 0.3s ease;
                margin: 1.5rem auto;
                display: block;
                width: 200px;
            }
            .stButton > button:hover {
                background: #2a4f73;
                transform: translateY(-2px);
            }
            div[data-testid="column"] {
                padding: 0 1rem;
            }
            [data-testid="stFileUploader"] {
                background: rgba(42, 79, 115, 0.2);
                border-radius: 6px;
                padding: 1rem;
            }
            [data-testid="stFileUploader"] > div > div {
                color: #4a90e2 !important;
            }
            .drag-text {
                color: #4a90e2;
                font-size: 0.9rem;
                text-align: center;
                margin-top: 0.5rem;
            }
            .file-limit-text {
                color: #6b7280;
                font-size: 0.8rem;
                text-align: center;
                margin-top: 0.25rem;
            }
            .output-field {
                background: rgba(74, 144, 226, 0.1);
                padding: 1rem;
                margin: 0.5rem 0;
                border-radius: 6px;
                border-left: 3px solid #4a90e2;
            }
            .field-label { 
                color: #4a90e2;
                font-weight: 500;
                font-size: 0.9rem;
                text-transform: uppercase;
            }
            .field-value { 
                color: white;
                font-size: 1.1rem;
                margin-top: 0.3rem;
            }
            .stSpinner > div {
                border-color: #4a90e2 !important;
                border-right-color: transparent !important;
            }
            div.stButton > button:first-child {
                border: none;
                background: linear-gradient(90deg, #1e3d59 0%, #2a4f73 100%);
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-title">Emirates Card Processor</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("macksofy_white (1).png")

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="upload-label">Upload Front Side</div>', unsafe_allow_html=True)
        front_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'], key="front")
        st.markdown('<div class="drag-text">Drag and drop your file here</div>', unsafe_allow_html=True)
        st.markdown('<div class="file-limit-text">Limit: 200MB per file • JPG, JPEG, PNG</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="upload-label">Upload Back Side</div>', unsafe_allow_html=True)
        back_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'], key="back")
        st.markdown('<div class="drag-text">Drag and drop your file here</div>', unsafe_allow_html=True)
        st.markdown('<div class="file-limit-text">Limit: 200MB per file • JPG, JPEG, PNG</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        process_button = st.button("Process Cards")

    if process_button:
        if front_file and back_file:
            with st.spinner("Processing your cards..."):
                front_s3_path = upload_to_s3(front_file, "front.jpg")
                back_s3_path = upload_to_s3(back_file, "back.jpg")
                
                if front_s3_path and back_s3_path:
                    try:
                        front_text = process_card_side(front_s3_path)
                        front_query = """Extract the following information from the given text as key-value pairs:
                        Full Name (list both given name and full name)
                        Card ID Number (format: ###-####-#######-#)
                        Date of Birth (format: DD/MM/YYYY)
                        Issue Date (format: DD/MM/YYYY)
                        Expiry Date (format: DD/MM/YYYY)
                        Please return only these five fields in JSON format."""
                        front_result = process_and_query(front_text, front_query)
                        
                        back_text = process_card_side(back_s3_path)
                        back_query = """Extract the following from the text as key-value pairs:
                        Occupation
                        Employer name (starting with 'Employer:')
                        Return these two fields in JSON format, ignoring any other text."""
                        back_result = process_and_query(back_text, back_query)
                        
                        try:
                            front_data = json.loads(front_result)
                            back_data = json.loads(back_result)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("<h3 style='color: #4a90e2; margin-bottom: 1rem;'>Front Side Details</h3>", unsafe_allow_html=True)
                                for key, value in front_data.items():
                                    st.markdown(
                                        f"""
                                        <div class="output-field">
                                            <div class="field-label">{key}</div>
                                            <div class="field-value">{value}</div>
                                        </div>
                                        """, 
                                        unsafe_allow_html=True
                                    )
                            
                            with col2:
                                st.markdown("<h3 style='color: #4a90e2; margin-bottom: 1rem;'>Back Side Details</h3>", unsafe_allow_html=True)
                                for key, value in back_data.items():
                                    st.markdown(
                                        f"""
                                        <div class="output-field">
                                            <div class="field-label">{key}</div>
                                            <div class="field-value">{value}</div>
                                        </div>
                                        """, 
                                        unsafe_allow_html=True
                                    )
                        except json.JSONDecodeError:
                            st.error("Error processing the card information")
                    except Exception as e:
                        st.error(f"Error during processing: {str(e)}")
                else:
                    st.error("Error uploading files")
        else:
            st.warning("Please upload both front and back images")

if __name__ == "__main__":
    main()