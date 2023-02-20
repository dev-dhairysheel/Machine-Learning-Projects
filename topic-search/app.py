import torch
import wikipedia
import transformers
import streamlit as st
from tokenizers import Tokenizer
from transformers import pipeline, Pipeline 


@st.cache(hash_funcs={Tokenizer: lambda _: None}, allow_output_mutation=True)
def load_qa_pipeline() -> Pipeline:
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    return qa_pipeline

@st.cache
def load_wiki_summary(query: str) -> str:
    results = wikipedia.search(query)
    return results

def answer_question(pipeline: Pipeline, question: str, paragraph: str) -> dict:
    input = {
        "question": question,
        "context": paragraph
    }
    output = pipeline(input)
    return output

if __name__ == "__main__":
    st.title("Topic Search")
    st.write("Search topic, Ask questions, Get Answers")

    topic = st.text_input("SEARCH TOPIC", "")

    question = st.text_input("QUESTION", "")

    if topic:
        summary = load_wiki_summary(topic)

        if question != "":
            qa_pipeline = load_qa_pipeline()
            result = answer_question(qa_pipeline, question, summary)
            answer = result["answer"]
            st.write(answer)
