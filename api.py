import os
import json
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Configure Gemini API
genai.configure(api_key="AIzaSyDrNErdtvDWfQzNXMTljil6Zbcy96Sdlr0")

# Pydantic models for request validation
class OutlineRequest(BaseModel):
    subject: str
    pages: int

class ContentRequest(BaseModel):
    title: str
    description: str
    pages: int

@app.post("/generate-outline")
async def generate_outline(request: OutlineRequest):
    """Generate a paper outline with sections based on subject and page count"""
    try:
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
        }

        model = genai.GenerativeModel(
            model_name="gemini-exp-1114",
            generation_config=generation_config,
            system_instruction="""You are a paper-writing assistant. When provided with a subject, the topic of the paper, and the desired number of pages, you must output a JSON-formatted response. The response should be a list where each list item represents a section of the paper. Each section must include its title, a brief description, and the number of pages allocated to it. Ensure that the sum of the pages matches the total requested. For example:

json
{
  "sections": [
    {
      "title": "Introduction",
      "description": "Provides an overview of the topic and outlines the purpose of the paper.",
      "pages": 1
    },
    {
      "title": "Subject One",
      "description": "Discusses the first key aspect of the topic in detail.",
      "pages": 3
    },
    {
      "title": "Subject Two",
      "description": "Explores another significant aspect, expanding on its implications.",
      "pages": 2
    }
  ]
}
use exactly this format.
Focus on clarity, relevance, and a logical structure for the paper."""
        )

        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(f"Create a {request.pages} page academic paper outline about {request.subject}")
        return json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-section")
async def generate_section(request: ContentRequest):
    """Generate content for a specific section of the paper"""
    try:
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
            model_name="gemini-exp-1114",
            generation_config=generation_config,
            system_instruction="You will receive a title, a description, and the number of pages required for specific sections of a paper. Your task is to write the assigned section of the paper based on the provided information, ensuring it aligns with the description and meets the specified page requirement."
        )

        chat_session = model.start_chat(history=[])
        prompt = f"Title: {request.title}\nDescription: {request.description}\nPages: {request.pages}\n\nWrite this section of the paper."
        response = chat_session.send_message(prompt)
        return {"content": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
