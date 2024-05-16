from flask import Flask, render_template,jsonify,request
from flask_cors import CORS
import requests,openai,os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
llm = AzureChatOpenAI(
	openai_api_version="2024-02-01",
	azure_deployment="gpt-4",
)
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data', methods=['POST'])
def get_data():
    app.logger.info('###-----> get data called')
    data = request.get_json()
    text=data.get('data')
    user_input = text
    try:
        conversation = ConversationChain(llm=llm,memory=memory)
        app.logger.info('###-------> try conversationChain')
        output = conversation.predict(input=user_input)
        app.logger.info('###-------> output finished')
#        memory.save_context({"input": user_input}, {"output": output})
        app.logger.info('###---------> saved in memory')
        return jsonify({"response":True,"message":output})
    except Exception as e:
        print(e)
        error_message = f'Error: {str(e)}'
        return jsonify({"message":error_message,"response":False})
    
if __name__ == '__main__':
    app.run()