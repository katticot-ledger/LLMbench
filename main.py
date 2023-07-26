import time
from enum import Enum
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp, OpenAI
from constants import CHROMA_SETTINGS
from utils import load_environment_variables, list_all
import streamlit as st
import pandas as pd

model_time = {}


class ModelType(Enum):
    OPEN_AI4 = "OpenAI 4"
    LLAMA_CPP = "LlamaCpp"
    GPT_4_ALL = "GPT4All"
    OPEN_AI = "OpenAI"


def get_selected_model(model_type, model_choice, model_n_ctx=2000):
    """Get selected language model."""
    model_classes = {
        ModelType.LLAMA_CPP: LlamaCpp,
        ModelType.GPT_4_ALL: GPT4All,
        ModelType.OPEN_AI4: lambda: OpenAI(temperature=0.9, model_name='gpt-3.5-turbo-0301'),
        ModelType.OPEN_AI: lambda: OpenAI(temperature=0.9, model_name='gpt-4-0613'),
    }
    model_class = model_classes.get(model_type)
    if model_class is None:
        st.write(f"Model {model_type} not supported!")
        exit()
    if model_type in [ModelType.LLAMA_CPP, ModelType.GPT_4_ALL]:
        model_path = f"models/{model_type.value}/{model_choice}"
        return model_class(model_path=model_path, n_ctx=model_n_ctx, verbose=False)
    else:
        return model_class()


def execute_query(query, qa):
    """Execute a query and return the results."""
    start_time = time.time()
    results = qa(query)
    end_time = time.time()
    execution_time = end_time - start_time
    return results, execution_time

def display_results(query, results, execution_time):
    """Display the results of a query."""
    answer, docs = results['result'], results['source_documents']
    st.header("Question:")
    st.caption(query)
    st.header("Answer:")
    st.caption(answer)
    for document in docs:
        expander = st.expander(
            f"\n> {document.metadata['source'].replace('source_documents/', '')}:")
        expander.write(document.page_content)
    st.write(f"Query took {execution_time} seconds.")


def setup_model(env, models):
    """Set up the model based on user input."""
    model_choice = None
    with st.sidebar:
        model_type = st.radio('Choose a model type:', [
                              model_type.value for model_type in ModelType])
        if model_type == ModelType.GPT_4_ALL.value or model_type == ModelType.LLAMA_CPP.value:
            model_choice = st.selectbox(
                "Choose a model:", models[model_type])  # Dropdown selector
            st.write(f"You selected {model_choice}")
    llm = get_selected_model(ModelType(model_type),
                             model_choice, env["model_n_ctx"])
    return llm, model_choice

def main():
    env = load_environment_variables()
    embeddings = HuggingFaceEmbeddings(model_name=env["embeddings_model_name"])
    db = Chroma(persist_directory=env["persist_directory"],
                embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(
        search_kwargs={"k": env["target_source_chunks"]})
    models = list_all(env["models_path"])
    llm, model_choice = setup_model(env, models)

    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    query = st.text_input("Enter a query:")
    if query:
        results, execution_time = execute_query(query, qa)
        display_results(query, results, execution_time)
        if model_choice not in st.session_state:
            st.session_state[model_choice] = []
        st.session_state[model_choice].append(execution_time)
        max_length = max(len(arr) for arr in st.session_state.values())
        df = pd.DataFrame.from_dict(st.session_state, orient='index')
        df.columns = [f'Time {i+1}' for i in range(max_length)]
        st.write(df)


if __name__ == "__main__":
    main()
