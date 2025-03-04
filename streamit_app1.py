from sentence_transformers import SentenceTransformer
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from PyPDF2 import PdfReader  # Import thư viện đọc PDF

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Sử dụng SentenceTransformer thay vì OpenAIEmbeddings
class CustomEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents):
        return self.model.encode(documents, convert_to_tensor=True).tolist()

    def embed_query(self, query):
        return self.model.encode(query, convert_to_tensor=True).tolist()


def generate_response(input_text, embedding_model):
    # Vector store
    db = Chroma(embedding_function=embedding_model)
    
    # Tìm kiếm trong vector database
    results = db.similarity_search_with_relevance_scores(input_text, k=3)
    
    if len(results) == 0 or results[0][1] < 0.7:
        st.info("Không tìm thấy kết quả nào")      
    else:
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=input_text)

        model = ChatOpenAI(temperature=0.7, api_key=openai_api_key)
        st.info(model.invoke(prompt).content)


# Page title
st.title("HOÀI BẢO ĐẸP TRAI - RAG")

# File upload: chỉ chấp nhận file PDF
uploaded_file = st.file_uploader('Upload your file:', type='pdf')

if st.button("Load Data"):
    if uploaded_file is not None:
        # Sử dụng PyPDF2 để đọc nội dung file PDF
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        documents = [text]
        
        # Chia nhỏ văn bản
        text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=100)
        chunks = text_splitter.create_documents(documents)

        # Dùng mô hình sentence-transformers để tạo embeddings
        embedding_model = CustomEmbeddings()

        # Tạo vector store từ dữ liệu
        db = Chroma.from_documents(chunks, embedding_model)
        st.success("Data Load OK")

with st.form("my_form"):
    text = st.text_area("Enter text:", "Bạn muốn hỏi gì?")
    submitted = st.form_submit_button("Submit", disabled=not uploaded_file)

    if submitted:
        generate_response(text, embedding_model)
