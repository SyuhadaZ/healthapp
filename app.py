import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
openai.api_key =  st.secrets["mykey"]

# Load data and embeddings (with error handling)
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['Question_Embedding'] = df['Question_Embedding'].apply(eval)  # Convert string to list/array
        return df
    except FileNotFoundError:
        st.error(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        return None

df = load_data("/content/qa_dataset_with_embeddings.csv")

if df is None:  # Stop execution if data loading failed
    st.stop()


# Load embedding model (cached)
@st.cache_resource
def load_model(model_name):
    return SentenceTransformer(model_name)

model = load_model('all-mpnet-base-v2')  # Or any other suitable model

# Streamlit app
st.title("Heart, Lung, and Blood Health Q&A")

user_question = st.text_area("Enter your question:", height=150)
similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.6, 0.05) # Allow user to adjust

if st.button("Get Answer"):
    if user_question:
        with st.spinner("Finding the most relevant answer..."):
            question_embedding = model.encode(user_question)
            similarities = cosine_similarity([question_embedding], np.array(df['Question_Embedding'].tolist()))
            most_similar_index = np.argmax(similarities)
            similarity_score = similarities[0][most_similar_index]

            if similarity_score >= similarity_threshold:
                answer = df.iloc[most_similar_index]['Answer']
                st.subheader("Answer:")
                st.write(answer)
                st.write(f"**Similarity Score:** {similarity_score:.2f}")  # Display score
            else:
                st.write("I apologize, but I don't have information on that topic yet. Could you please ask other questions?")
    else:
        st.warning("Please enter a question.")

if st.button("Clear"):
    user_question = "" # Clear the text area
    st.experimental_rerun() # Rerun streamlit to clear the input
