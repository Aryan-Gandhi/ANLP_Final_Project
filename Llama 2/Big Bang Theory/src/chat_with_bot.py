# chat_with_bot.py
from transformers import pipeline, AutoTokenizer
import config

# Function to chat with the bot
def chat_with_bot(model, tokenizer):
    text_generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    conversation_history = ""

    while True:
        user_input = input(">> User: ")
        if user_input.lower() == 'quit':
            break

        model_input = conversation_history + f"<s>[INST] {user_input} [/INST]"
        response = text_generator(model_input)
        bot_response = response[0]['generated_text'].strip()
        print("Model Bot: " + bot_response)
