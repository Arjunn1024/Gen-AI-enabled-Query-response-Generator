# Question-Answer Bot

This project is a Question-Answer Bot built using various NLP models, including T5 for question generation and spaCy for text processing. The bot processes text from uploaded PDF files, generates questions, extracts answers from the text, and evaluates the questions using ROUGE scores.

## Features
- **PDF Text Extraction**: Extracts text content from uploaded PDF files.
- **Question Generation**: Generates questions based on the extracted text using a pre-trained T5 model.
- **Answer Extraction**: Extracts answers from the text based on the generated questions.
- **ROUGE Score Calculation**: Evaluates the generated questions by calculating ROUGE scores against reference questions.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/<your-username>/question-answer-bot.git
    cd question-answer-bot
    ```

2. **Create and activate a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    - Create a `.env` file in the root directory of the project.
    - Add your Google API key to the `.env` file:
      ```env
      GOOGLE_API_KEY=your_google_api_key
      ```

## Usage

1. **Run the Streamlit app**:
    ```sh
    streamlit run app.py
    ```

2. **Upload PDF Files**:
    - Use the sidebar to upload one or more PDF files.

3. **Process the PDFs**:
    - Click the "Submit & Process" button to extract text, generate questions, extract answers, and calculate ROUGE scores.

4. **View Results**:
    - The generated questions, extracted answers, and ROUGE scores will be displayed on the main page.

## Code Overview

- **`app.py`**: The main file for running the Streamlit app. It includes functions for extracting text from PDFs, generating questions, extracting answers, and calculating ROUGE scores.
- **`requirements.txt`**: Lists the dependencies required to run the project.
- **`.env`**: Environment file to store your Google API key.

## Dependencies

- streamlit
- pdfplumber
- transformers
- spacy
- rouge-score
- langchain
- python-dotenv

## Example

Here is an example of the bot in action:

1. **Upload PDF**:
    ![Upload PDF](https://example.com/upload_screenshot.png)

2. **View Generated Questions and Answers**:
    ![Generated Q&A](https://example.com/generated_qa_screenshot.png)

3. **View ROUGE Scores**:
    ![ROUGE Scores](https://example.com/rouge_scores_screenshot.png)



