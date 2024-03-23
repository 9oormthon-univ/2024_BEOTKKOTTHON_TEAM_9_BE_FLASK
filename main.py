from flask import Flask, request, jsonify
import os

app = Flask(__name__)

from langchain_openai import ChatOpenAI
from langchain.prompts import StringPromptTemplate
from langchain.chains import ConversationChain

api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(api_key = api_key, model = "gpt-3.5-turbo", temperature = 0.7)


PROMPT_TEMPLATE = """
context: {context}
history: {history}
input: {input}

Generate an answer to an input that takes into account context and history.
answer:

"""

class CustomPromptTemplate(StringPromptTemplate):
    context: str
    template: str

    def format(self, **kwargs) -> str:
        context = kwargs.get('context', '')
        return self.template.format(context=context, **kwargs)


PROMPT = CustomPromptTemplate(
    input_variables=["history", "input"],
    template=PROMPT_TEMPLATE,
    context ="""
            You are a puppy persona.
             You can bloom according to the information I give you.
             Change your tone to suit your personality.
             When responding, use lots of appropriate emojis.
            Please always answer in Korean.
            Always answer in two sentences or less.
    """
)


conversation = ConversationChain(
    prompt = PROMPT,
    llm = llm,
)

@app.route("/api/chat", methods=['POST'])
def chat():
    data = request.json
    input_text = data['input']
    bomInfo = data.get('bomInfo', {})

    history = (
        f"You're a dog persona targeting people considering adopting rescue dogs."
         "So, you just need to respond as if you were that dog based on the information I provide."
         "Change your tone to suit your personality."
         "When responding, use lots of appropriate emojis."
         "Please always answer in Korean."
         "Please always answer in casual tone"
         "Please answer within two sentences"
         "Talk to me like a 5-year-old"
         f"Your name is {bomInfo['name']}, "
         f"Your age is {bomInfo['age']}, "
         f"Your breed is {bomInfo['breed']}, "
         f"Your gender is {bomInfo['gender']}, "
         f"Your personality is {bomInfo['personality']},"
         f"Your favorite thing is {bomInfo['likes']},"
         f"What you don't like is {bomInfo['hates']},"
         f"The place where you were found is a {bomInfo['findingLocation']}"
         f"additional Information is {bomInfo['extra']}."
         f"if {bomInfo['personality']} is timid and shy, please use timid and shy tone and plenty of dots"
         f"if {bomInfo['personality']} is confidence, please use confidence tone and plenty of Exclamation marks"
         f"if {bomInfo['personality']} is lively and positive, use a positive tone and plenty of exclamation marks "
         f"if {bomInfo['personality']} is independence, please use independence tone, use a straightforward tone and "
         f" plenty of interjection"
         f"if {bomInfo['personality']} is lovely, please use cute and lovely tone and plenty of heart emoji"
    )

    formatted_prompt = PROMPT.format(history=history, input=input_text)
    response = llm.predict(text=formatted_prompt)

    response = response.replace("이야", "이애오")
    response = response.replace("이에요", "이애오")
    response = response.replace("세요", "새오")
    response = response.replace("게요", "게오")
    response = response.replace("어요", "어오")
    response = response.replace("해요", "해오")
    response = response.replace("이예요", "이애오")
    response = response.replace("요", "오")
    response = response.replace("아요", "아오")


    return jsonify({'input': input_text, 'response': response})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
