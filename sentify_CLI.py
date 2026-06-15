import os 
from google import genai
from dotenv import load_dotenv

# Load the hidden .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Setup the Gemini client safely
client = genai.Client(api_key=api_key)

#2 - Read the reviews from your text file :
try:
    with open("reviews.txt", "r", encoding="utf-8") as file:
        customer_reviews = file.read()
except FileExistsError:
    print("Error: The file reviews.txt was not found!")
    exit()

#3 - create a powerful prompt for gemini
prompt = f"""
You are an expert business consultant. Analyze the following customer reviews.
Provide a professional response with three specific sections:
1. **Compliments**: Bullet points of what customers like.
2. **Complaints**: Bullet points of what customers dislike.
3. **Actionable Business Advice**: 2-3 specific recommendations for the business owner to improve their sales and fix these issues.

Here are the reviews:
{customer_reviews}
"""

print("Analyzing reviews with Gemini AI... Please wait...")

#4 - send the request to the gemini model 
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=prompt,
)

#5 - Print the professional report to your terminal 

print("\n=== AI BUSINESS INSIGHTS REPORT ===")
print(response.text)