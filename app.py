import streamlit as st
from backend import rag_backend as backend

st.set_page_config(page_title="My RAG Project")

new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;"> Indias Rich </p>'
st.markdown(new_title, unsafe_allow_html=True)

if 'vector_index' not in st.session_state:
    with st.spinner("WAIT for sometime !! :)"):
        st.session_state.vector_index = backend.get_document_index()
        
input_text =  st.text_area("Input text", label_visibility="collapsed")
go_button = st.button("Submit!", type="primary")

if go_button:
    with st.spinner("Trying to fetch the response"):
        response_content = backend.doc_rag_response(index=st.session_state.vector_index, prompt=input_text)
        st.write(response_content)