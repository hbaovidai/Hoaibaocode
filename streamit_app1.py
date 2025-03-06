from sentence_transformers import SentenceTransformer
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import Document
from PyPDF2 import PdfReader

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

class CustomEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents):
        return self.model.encode(documents, convert_to_tensor=True).tolist()

    def embed_query(self, query):
        return self.model.encode(query, convert_to_tensor=True).tolist()

def process_pdf(uploaded_file, embedding_model):
    pdf_reader = PdfReader(uploaded_file)
    text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    
    text_splitter = CharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        separator="\n"
    )
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    db = Chroma.from_documents(documents, embedding_model)
    return db

def generate_response(input_text, openai_api_key):
    if "db" not in st.session_state:
        st.error("Vui lòng upload file PDF trước")
        return
    
    db = st.session_state.db
    results = db.similarity_search_with_relevance_scores(input_text, k=3)
    
    if not results or results[0][1] < 0.7:
        st.info("Không tìm thấy thông tin phù hợp trong tài liệu")
        return
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=input_text)
    
    try:
        model = ChatOpenAI(temperature=0.7, api_key=openai_api_key)
        response = model.invoke(prompt)
        st.success(response.content)
    except Exception as e:
        st.error(f"Lỗi kết nối OpenAI: {str(e)}")

# Main app
st.title("HỆ THỐNG HỖ TRỢ TƯ VẤN - RAG")

# Sidebar config
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
embedding_model = CustomEmbeddings()

# File upload
uploaded_file = st.file_uploader('Tải lên tài liệu PDF', type='pdf')
if uploaded_file and "db" not in st.session_state:
    with st.spinner("Đang xử lý tài liệu..."):
        st.session_state.db = process_pdf(uploaded_file, embedding_model)
        st.success("Đã sẵn sàng hỏi đáp!")

# Q&A form
with st.form("qa_form"):
    question = st.text_area("Câu hỏi của bạn", "Bạn muốn hỏi gì?")
    submitted = st.form_submit_button("Gửi câu hỏi", 
                                    disabled=not (uploaded_file and openai_api_key))
    
    if submitted:
        if not openai_api_key.startswith("sk-"):
            st.error("Vui lòng nhập API key hợp lệ")
        else:
            with st.spinner("Đang tìm câu trả lời..."):
                generate_response(question, openai_api_key)
