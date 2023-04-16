"""Main entrypoint for the app."""

import logging
import pickle
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import json
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")

list_prompts = ["practice_affirmation",
"guided_journaling",
"practice_breathing",
"filler or counselling",
"practice_awareness",
"send_love_to_yourself"]

prompt = """
Imagine a patient is experiencing discomfort due to feelings of self-doubt and low self-esteem. A counselor may provide some advice and suggest the most relevant tools from the following list: ["practice_affirmation", "guided_journaling", "practice_breathing", "filler or counseling", "practice_awareness", "send_love_to_yourself"].

Please generate a response in the following JSON format:

{
"counseling": "{}",
"tools": [
{
"name": "name of the suggested tool",
"prompt": "a detailed prompt or instruction for the user to follow",
"meta": "additional information required for specific tools (if applicable)"
}
],
"message": "a warm, closing message for the patient"
}

For the following tools, please include the specified meta tags:

practice_breathing: Include a meta tag object containing one relevant suggested breathing exercise from the following options: [5-5-5-5, 4-7-8, 4-5-6, 4-4].
practice_affirmation: Include a meta tag object containing a list of all the affirmative statements used in the prompt.
practice_awareness: Include a meta tag object containing the corresponding awareness number used in the prompt.

Please generate a detailed counseling plan for the patient, considering their feelings of self-doubt and low self-esteem, and provide specific prompts for the selected tools.
"""

class InputData(BaseModel):
    data: str

class GenResponse(BaseModel):
    message: str

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    logging.info("loading docsearch")
    if not Path("docsearch.pkl").exists():
        raise ValueError("docsearch.pkl does not exist. Make sure you run ingest.py")
    with open("docsearch.pkl", "rb") as f:
        global vectorstore 
        vectorstore = pickle.load(f)

@app.get("/")
async def get(request: Request):
    return {'message': 'Hello World! This is Being AI ~~~'} 

@app.post('/gen')
async def gen_endpoint(input_data: InputData):

    while True:
        try:
            chain = load_qa_chain(ChatOpenAI(openai_api_key=api_key), chain_type="stuff")
            query = f"{input_data.data}, {prompt}"
            docs = vectorstore.similarity_search(query)
            resp = chain.run(input_documents=docs, question=query)
            jsonify = json.loads(resp)
            return JSONResponse(jsonify)

        except Exception as e:
            logging.error(e)
            resp = GenResponse(message="Oops. Something went wrong. Try again.")
            return JSONResponse(content=resp)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)