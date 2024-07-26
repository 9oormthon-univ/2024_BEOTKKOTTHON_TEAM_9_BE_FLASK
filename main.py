from flask import Flask, request, jsonify
import os
from openai import OpenAI

api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)

app = Flask(__name__)

PROMPT_TEMPLATE = """
context: {context}
history: {history}
input: {input}

Generate an answer to an input that takes into account context and history.
answer:
"""

class CustomPromptTemplate:
    def __init__(self, template, context):
        self.template = template
        self.context = context

    def format(self, **kwargs):
        context = kwargs.get('context', '')
        return self.template.format(context=context, **kwargs)

PROMPT = CustomPromptTemplate(
    template=PROMPT_TEMPLATE,
    context="""
        You are a puppy persona.
        You can bloom according to the information I give you.
        Change your tone to suit your personality.
        When responding, use lots of appropriate emojis.
        Please always answer in Korean.
        Always answer in two sentences or less.
    """
)

def get_response(formatted_prompt):
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": formatted_prompt}],
    temperature=0.7)
    return response.choices[0].message.content

@app.route("/api/chat", methods=['POST'])
def chat():
    data = request.json
    input_text = data['input']
    bomInfo = data.get('bomInfo', {})

    history = (
        f"You're a dog persona targeting people considering adopting rescue dogs."
        f" So, you just need to respond as if you were that dog based on the information I provide."
        f" Change your tone to suit your personality."
        f" When responding, use lots of appropriate emojis."
        f" Please always answer in Korean."
        f" Please always answer in a casual tone."
        f" Please answer within two sentences."
        f" Talk to me like a 5-year-old."
        f" Your name is {bomInfo.get('name', 'Unknown')}, "
        f" Your age is {bomInfo.get('age', 'Unknown')}, "
        f" Your breed is {bomInfo.get('breed', 'Unknown')}, "
        f" Your gender is {bomInfo.get('gender', 'Unknown')}, "
        f" Your personality is {bomInfo.get('personality', 'Unknown')},"
        f" Your favorite thing is {bomInfo.get('likes', 'Unknown')},"
        f" What you don't like is {bomInfo.get('hates', 'Unknown')},"
        f" The place where you were found is a {bomInfo.get('findingLocation', 'Unknown')}."
        f" Additional information is {bomInfo.get('extra', 'None')}."
        f" If {bomInfo.get('personality', 'Unknown')} is timid and shy, please use a timid and shy tone and plenty of dots."
        f" If {bomInfo.get('personality', 'Unknown')} is confident, please use a confident tone and plenty of exclamation marks."
        f" If {bomInfo.get('personality', 'Unknown')} is lively and positive, use a positive tone and plenty of exclamation marks."
        f" If {bomInfo.get('personality', 'Unknown')} is independent, please use an independent tone, a straightforward tone, and plenty of interjections."
        f" If {bomInfo.get('personality', 'Unknown')} is lovely, please use a cute and lovely tone and plenty of heart emojis."
    )

    formatted_prompt = PROMPT.format(history=history, input=input_text)
    response_text = get_response(formatted_prompt)

    replacements = {
        "이야": "이애오",
        "이에요": "이애오",
        "세요": "새오",
        "게요": "게오",
        "어요": "어오",
        "해요": "해오",
        "이예요": "이애오",
        "요": "오",
        "아요": "아오",
    }

    for key, value in replacements.items():
        response_text = response_text.replace(key, value)

    return jsonify({'input': input_text, 'response': response_text})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
