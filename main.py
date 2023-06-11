import time
from enum import Enum
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
from constants import CHROMA_SETTINGS
from utils import load_environment_variables, list_all
import streamlit as st
import pandas as pd

model_time = {}


class ModelType(Enum):
    LLAMA_CPP = "LlamaCpp"
    GPT_4_ALL = "GPT4All"


def get_selected_model(model_type, model_path, model_n_ctx):
    """Get selected language model."""
    if model_type == ModelType.LLAMA_CPP:
        return LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, verbose=False)
    elif model_type == ModelType.GPT_4_ALL:
        return GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', verbose=False)
    else:
        st.write(f"Model {model_type} not supported!")
        exit()


def execute_query(query, qa, model):
    """Execute a query and return the results."""
    start_time = time.time()
    results = qa(query)
    end_time = time.time()
    model_time[model] = (end_time - start_time)
    answer, docs = results['result'], results['source_documents']
    st.header("Question:")
    st.caption(query)
    st.header("Answer:")
    st.caption(answer)
    for document in docs:
        expander = st.expander(
            "\n> " + document.metadata["source"].replace('source_documents/', '') + ":")
        expander.write(document.page_content)
    st.write(f"Query took {end_time - start_time} seconds.")
    return model_time[model]


def main():
    env = load_environment_variables()
    models = list_all(env["models_path"])
    embeddings = HuggingFaceEmbeddings(model_name=env["embeddings_model_name"])
    db = Chroma(persist_directory=env["persist_directory"],
                embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(
        search_kwargs={"k": env["target_source_chunks"]})
    selected_file = models

    with st.sidebar:
        model_type = st.radio('Choose a model type:', [str(
            key) for key in models.keys()])  # Radio button selector
        model_choice = st.selectbox(
            "Choose a model:", selected_file[model_type])  # Dropdown selector

    st.write(f"You selected {model_choice}")

    llm = get_selected_model(ModelType(
        model_type), "models/"+model_type+"/"+model_choice, env["model_n_ctx"])

    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    query = st.text_input("Enter a query:")
    if query:
        execution_time = execute_query(query, qa, model_choice)
        if model_choice not in st.session_state:
            st.session_state[model_choice] = []
        st.session_state[model_choice].append(execution_time)
        max_length = max(len(arr) for arr in st.session_state.values())
        df = pd.DataFrame.from_dict(st.session_state, orient='index')
        df.columns = ['Time {}'.format(i+1) for i in range(max_length)]
        st.write(df)


if __name__ == "__main__":
    main()
