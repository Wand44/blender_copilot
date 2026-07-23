#from ollama import chat, ChatResponse
import json 
import random

NUMBER_OF_QUESTIONS = 2

training_data = []

with open("training_data.jsonl", 'r') as f:
        for line in f:
            training_data.append(json.loads(line.strip()))

# take a few random questions from the training data 
questions = random.choices(training_data, k = NUMBER_OF_QUESTIONS )

with open('test.jsonl', 'w') as f:
    for item in questions:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

# Ask the ORIGINAL model to answer them
"""
def get_llm_response():
    response: ChatResponse = chat(model=LLM_MODEL, messages=[
    {   'role': 'user',
        'content': combine_contents()}])
    return response.message.content

# Ask the NEW model to answer them

# Compare the answers. Output a .md with a table of question, original answer, new answer, and a quality score (1-5) for the new answer.


"""