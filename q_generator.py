import streamlit as st
import pdfplumber
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from rouge_score import rouge_scorer
import spacy
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize spaCy model
nlp = spacy.load("en_core_web_sm")

# Load model and tokenizer for T5
model_name = 'valhalla/t5-small-e2e-qg'
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Function to extract text from PDF
def extract_text_from_pdf(pdf):
    with pdfplumber.open(pdf) as pdf_file:
        text = ""
        for page in pdf_file.pages:
            text += page.extract_text()
    return text

# Function to generate questions from text
def generate_questions_from_text(text, model, tokenizer, num_questions=5, start_phrases=["how", "which", "do"]):
    generated_questions = []
    for start_phrase in start_phrases:
        input_text = f"{start_phrase}: {text}"
        inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)
        outputs = model.generate(inputs['input_ids'],
                                 max_length=100,
                                 num_return_sequences=num_questions,
                                 num_beams=5,
                                 no_repeat_ngram_size=2,
                                 early_stopping=True)
        questions = list(set([tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]))
        generated_questions.extend(questions)
    return generated_questions[:num_questions]

# Function to format and remove duplicate questions
def format_and_remove_duplicates(generated_questions):
    seen_questions_global = set()
    formatted_questions = []

    for i, question_set in enumerate(generated_questions, start=1):
        question_segments = question_set.split('<sep>')
        questions = []
        for segment in question_segments:
            questions.extend([q.strip() + '?' for q in segment.split('?') if q.strip()])

        unique_questions_local = []
        seen_questions_local = set()

        for q in questions:
            if q not in seen_questions_local:
                unique_questions_local.append(q)
                seen_questions_local.add(q)

        # Add unique questions from the local set to the global set
        for q in unique_questions_local:
            if q not in seen_questions_global:
                formatted_questions.append(f"{q}")
                seen_questions_global.add(q)

    return formatted_questions

# Function to split text into sentences
def split_into_sentences(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences

# Function to calculate ROUGE scores
def calculate_rouge(reference_questions, formatted_questions):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = []

    for i, gen_q in enumerate(formatted_questions):
        reference = reference_questions[i][0] if i < len(reference_questions) else reference_questions[-1][0]
        score = scorer.score(reference, gen_q)
        scores.append(score)

    return scores

# Streamlit app
def main():
    st.set_page_config(page_title="Question-Answer Bot")
    st.title("Question-Answer Bot")

    if "question_answer_pairs" not in st.session_state:
        st.session_state.question_answer_pairs = []

    if "rouge_scores" not in st.session_state:
        st.session_state.rouge_scores = []

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                for pdf in pdf_docs:
                    raw_text = extract_text_from_pdf(pdf)
                    sentences = split_into_sentences(raw_text)
                    generated_questions = generate_questions_from_text(raw_text, model, tokenizer, num_questions=5)
                    formatted_questions = format_and_remove_duplicates(generated_questions)
                    st.session_state.question_answer_pairs.extend(formatted_questions)

                reference_questions = [
                    ["What is immortality within human reach?"],
                    ["By what century is the prospect of living up to 5000 years might well become reality?"],
                    ["What does the idea that death is a key to life are at best based on dubious science?"],
                    ["What do Chipko activists in TehriGarhwal sing praising their hills as paradise, the place of Gods, where the mountains bloom with rare plants and dense cedars?"],
                    ["What was the name of?"],
                    ["By what century is the prospect of living up to 5000 years based on dubious science?"],
                    ["What does the scientific fraternity rarely take seriously?"],
                    ["What did Chipko activists sing in the 1970s?"],
                    ["What was the name of the movement to save the indigenous forests of oak and rhododendron from being felled by the Forest Department?"],
                    ["What was the name of the movement to save the indigenous forests of oak and rhododendron?"],
                    ["What does the idea that death is key to life are at best based on?"],
                    ["What do Chipko activists in TehriGarhwal sing?"],
                    ["What did ChipKo protest against?"]
                ]

                st.session_state.rouge_scores = calculate_rouge(reference_questions, st.session_state.question_answer_pairs)
                st.success("Done")

    st.header("Generated Questions")
    for i, question in enumerate(st.session_state.question_answer_pairs, 1):
        st.subheader(f"Question {i}: {question}")
        if i <= len(st.session_state.rouge_scores):
            rouge_score_str = f"ROUGE score for Question {i}: {st.session_state.rouge_scores[i-1]}"
            st.markdown(f"<span style='color:yellow;'>{rouge_score_str}</span>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
