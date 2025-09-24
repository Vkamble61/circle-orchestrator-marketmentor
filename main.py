import sys
import os
from dotenv import load_dotenv


from src.market_mentor.crew import MarketMentor

def run():
    """ run the crew   """
    load_dotenv()  # Load environment variables from .env file
    
    print("Knowledge Driven Marketing Mentor...")
    #MarketMentor().crew().kickoff()

    input_docs = {'source': "src/market_mentor/knowledge_source/introduction-to-genai.pdf"}
    result = MarketMentor().crew().kickoff(inputs=input_docs)  # Start the crew process
    print("Final Result:", result)
    
if __name__ == "__main__":
    run()