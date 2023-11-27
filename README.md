# Comparative Analysis of LLMs for Character-Specific Chatbots: Case Studies with DialogGPT and LlaMA-2

Welcome to our research project focusing on a comparative analysis between two prominent large language models, DialogGPT and LlaMA-2, specifically applied to character-specific chatbots. Our study delves into key metrics such as perplexity, BLEU scores, fluency, and relevancy to provide a comprehensive understanding of the performance of these models.

## Project Setup

### Dependencies Installation
Before running the project, ensure that all dependencies are installed. Execute the following command:

```bash
pip install -r requirements.txt
```

### Folder Structure
The project is organized into two main folders: DialoGPT and LlaMA-2. Each folder contains subfolders for source code (`src`), datasets (`dataset`), and model files (`model`).

## Running the Project

### Trained Models
As the DialoGPT model files are substantial in size, they are not included in this repository. Please connect to Google Drive and use the provided links below to access the necessary model files:

- [**Big Bang Theory Model**](https://drive.google.com/drive/folders/1CBpYKxK1L1odNu6GLvd0RVuUtQa_K5Lb?usp=sharing)
- [**Rick and Morty Model**](https://drive.google.com/drive/folders/1BxKQ41QSZ6HOjCn0NVoerrzZhDKSfSse?usp=sharing)

Ensure that the "output-small" file is placed in the same directory as the `chat_with_bot.py` script.

### Execution Steps

1. To run the trained models, execute the `chat_with_bot.py` script in either the DialoGPT folder for either Rick and Morty or Big Bang Theory.
2. If running the entire code from scratch, ensure that your GPU is activated, then execute the `main.py` file.
3. Repeat the above steps for both the Rick and Morty and Big Bang Theory chatbots.

## Results and Evaluation

The results are presented in a comprehensive table, showcasing perplexity and BLEU scores for both models. Additionally, pie charts visually represent fluency and relevancy based on manual evaluations from a population of 500 people.

We hope this research contributes valuable insights to the field of character-specific chatbots, aiding developers and researchers in making informed decisions about model selection for their applications. Feel free to explore our findings and reach out for any further inquiries or discussions.
