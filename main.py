import streamlit as st
from contextual_answers import ContextualAnswers
from dotenv import load_dotenv

# Load environment variables - you need HUGGINGFACEHUB_API_TOKEN and OPENAI_API_KEY
load_dotenv()

# Set the page title in Streamlit
st.set_page_config(page_title='Contextual Answers')
# Set the title of the Streamlit app
st.title('Contextual Answers App')

# Upload a text file to the Streamlit app
uploaded_file = st.file_uploader('Upload your files', type=['txt'], accept_multiple_files=True)

combined_content = ""
for file in uploaded_file:
    file_content = file.getvalue().decode()
    combined_content += file_content


# Input a question to the Streamlit app. The input field is disabled until a file is uploaded.
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

# Initialize the ContextualAnswers object with the specified repository ID
# res = ContextualAnswers(repo_id="meta-llama/Llama-2-7b-hf") # not enough memory for this one
# res = ContextualAnswers(repo_id="tiiuae/falcon-7b") # gives uncertain answers
# res = ContextualAnswers(repo_id="bigscience/bloom-560m") # gives too short answers
# res = ContextualAnswers(repo_id="tiiuae/falcon-40b") # api call - time out
# res = ContextualAnswers(repo_id='tiiuae/falcon-7b-instruct') # false answers
# res = ContextualAnswers(repo_id='google/flan-t5-xxl')
res = ContextualAnswers()

# Initialize the chain with the specified type and uploaded file
res.init_chain(chain_type='stuff', uploaded_file=combined_content)

# Initialize an empty result
response = None

# Calculate the answer to the query
with st.spinner('Calculating...'):
    if res and query_text != '':
        # Get the answer to the query
        response = res.get_answer(query_text)

# If there is a result, display it in an info box
if response is not None:
    st.info(response.get('result'))
