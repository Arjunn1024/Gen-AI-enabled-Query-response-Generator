Question-Answer Bot
This project is a web application built with Streamlit that extracts text from PDF files, generates questions and answers using NLP models, and evaluates the quality of generated questions using ROUGE scores. The application also integrates with Google's Generative AI for advanced question answering.

Features
Extract text from PDF files.
Generate questions from the extracted text using the T5 transformer model.
Format and remove duplicate questions.
Extract answers from the text based on generated questions.
Calculate ROUGE scores to evaluate the quality of the generated questions.
Integrate with Google's Generative AI for detailed question answering.
Installation
Prerequisites
Python 3.7+
pip package manager
Clone the Repository
bash
Copy code
git clone https://github.com/your-username/question-answer-bot.git
cd question-answer-bot
Create and Activate a Virtual Environment
bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install Dependencies
bash
Copy code
pip install -r requirements.txt
Setup Environment Variables
Create a .env file in the root directory of the project and add your Google Generative AI API key:

env
Copy code
GOOGLE_API_KEY=your_google_api_key_here
Usage
Running the Application
bash
Copy code
streamlit run app.py
Using the Application
Upload one or more PDF files using the file uploader in the sidebar.
Click on the "Submit & Process" button to extract text, generate questions and answers, and calculate ROUGE scores.
View the generated question-answer pairs and their respective ROUGE scores on the main page.
Project Structure
app.py: Main application script.
requirements.txt: List of dependencies.
.env: Environment variables file (not included in the repository).
Functions
extract_text_from_pdf
Extracts text from a PDF file using pdfplumber.

generate_questions_from_text
Generates questions from the extracted text using the T5 transformer model.

format_and_remove_duplicates
Formats the generated questions and removes duplicates.

split_into_sentences
Splits the extracted text into sentences using spaCy.

extract_answers
Extracts answers from the text based on the generated questions.

calculate_rouge
Calculates ROUGE scores for evaluating the quality of generated questions.

get_conversational_chain
Sets up a conversational chain with Google's Generative AI for question answering.

Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
Streamlit
Hugging Face Transformers
pdfplumber
spaCy
Google Generative AI
