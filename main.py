import streamlit as st
from contextual_answers import ContextualAnswers
from dotenv import load_dotenv

# Load environment variables - you need HUGGINGFACEHUB_API_TOKEN for HuggingFaceEmbeddings
load_dotenv()

# Set the page title in Streamlit
st.set_page_config(page_title='Contextual Answers')
# Set the title of the Streamlit app
st.title('Contextual Answers App')

# Upload a text file to the Streamlit app
uploaded_file = st.file_uploader('Upload an article', type=['txt'])

# Input a question to the Streamlit app. The input field is disabled until a file is uploaded.
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

# Initialize the ContextualAnswers object with the specified repository ID
res = ContextualAnswers(repo_id="meta-llama/Llama-2-7b-hf")

# Initialize the chain with the specified type and uploaded file
res.init_chain(chain_type='stuff', uploaded_file=uploaded_file)

# Initialize an empty list to store the results
result = []

# Calculate the answer to the query
with st.spinner('Calculating...'):
    if res:
        # Get the answer to the query
        response = res.get_answer(query_text)
        result.append(response)

# If there is a result, display it in an info box
if result is not None:
    st.info(response)