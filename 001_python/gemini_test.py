from google import genai
import os

os.environ["GEMINI_API_KEY"] = "AIzaSyCv2CRufLmQHMiyKps4RpLQy3B1VF3Ga60"

# API 키는 환경 변수 GEMINI_API_KEY에서 자동 인식
client = genai.Client()


q = input("AI질문 : ")
response = client.models.generate_content(
    model="gemini-2.5-flash", 
    contents=q
)
print('-' * 60)
print(response.text)
print('-' * 60)