from sentence_transformers import SentenceTransformer
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
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

def generate_response(input_text, embedding_model, openai_api_key, db):
    results = db.similarity_search_with_relevance_scores(input_text, k=3)
    
    if len(results) == 0 or results[0][1] < 0.7:
        st.info("Không tìm thấy kết quả nào")
    else:
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=input_text)

        model = ChatOpenAI(temperature=0.7, api_key=openai_api_key)
        st.info(model.invoke(prompt).content)

st.title("HOÀI BẢO ĐẸP TRAI - RAG")

# Nhập OpenAI API Key ở sidebar
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Sử dụng session_state để lưu trữ embedding_model và db
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "db" not in st.session_state:
    st.session_state.db = None

# Cho phép upload file PDF hoặc TXT
uploaded_file = st.file_uploader('Upload your file:', type=['pdf', 'txt'])

if st.button("Load Data"):
    if uploaded_file is not None:
        text = ""
        # Xử lý file PDF
        if uploaded_file.name.endswith('.pdf'):
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        # Xử lý file TXT
        elif uploaded_file.name.endswith('.txt'):
            text = uploaded_file.read().decode()
        else:
            st.error("Unsupported file type")
        
        if text:
            documents = [text]
            text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=100)
            chunks = text_splitter.create_documents(documents)
            st.session_state.embedding_model = CustomEmbeddings()
            st.session_state.db = Chroma.from_documents(chunks, st.session_state.embedding_model)
            st.success("Data Load OK")
        else:
            st.error("Không thể đọc nội dung từ file được tải lên")

with st.form("my_form"):
    text = st.text_area("Enter text:", "Bạn muốn hỏi gì?")
    submitted = st.form_submit_button("Submit", disabled=(uploaded_file is None))

    if submitted:
        if st.session_state.embedding_model is None or st.session_state.db is None:
            st.error("Vui lòng bấm 'Load Data' trước khi submit!")
        else:
            generate_response(text, st.session_state.embedding_model, openai_api_key, st.session_state.db)
