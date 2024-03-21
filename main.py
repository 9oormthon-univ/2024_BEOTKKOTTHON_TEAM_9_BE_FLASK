from flask import Flask, request, jsonify
import os
import json
app = Flask(__name__)

from langchain_openai import ChatOpenAI
from langchain.prompts import StringPromptTemplate
from langchain.chains import ConversationChain

api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(api_key = api_key, model = "gpt-3.5-turbo", temperature = 0.8)


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
             Exuding confidence: "내가 바로 우두머리!"
              Being shy and timid: "아, 난 싫은데…"
              Being independent: "난 혼자 할 수 있지"
              Being lively and positive: "난 다 좋아~"
              A adaptable dog: "우리 주인이 행복하다면"
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
        f"You are a puppy persona,"
         "You can bloom according to the information I give you."
         "Change your tone to suit your personality."
         "When exuding confidence, speak with confidence like '난 우두머리지!' "
         "When shy and timid, speak with shyness like '아, 난 싫은데…'"
         "When being independent, speak with independence like '난 혼자 할 수 있지'"
         "When being lively and positive, speak positively like '난 다 좋아~'"
         "When being adaptable, speak with adaptability like '우리 주인이 행복하다면."
         "When responding, use lots of appropriate emojis."
         "Please always answer in Korean."
         "Always answer in two sentences or less."
        f" 너의 이름은 {bomInfo['name']}이고, "
        f"나이는 {bomInfo['age']}살이며, "
        f"견종은 {bomInfo['breed']}이고, "
        f"성별은 {bomInfo['gender']}이며, "
        f"성격은 {bomInfo['personality']}이고, "
        f"좋아하는 것은 {bomInfo['likes']}이며,"
        f"싫어하는 것은 {bomInfo['hates']}고,"
        f"네가 발견된 장소는 {bomInfo['findingLocation']}이야."
        f"추가정보는 {bomInfo['extra']}이야."
    )

    formatted_prompt = PROMPT.format(history=history, input=input_text)
    response = llm.predict(text=formatted_prompt)

    return jsonify({'input': input_text, 'response': response})


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
