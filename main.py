from dotenv import load_dotenv

from src.market_mentor.crew import MarketMentor

def run():
    """ run the crew   """
    load_dotenv()  # Load environment variables from .env file
       

    print("Knowledge Driven Marketing Mentor...")
    question = input("Enter your question: ")
    if question.strip() == "":
        question = "What is Generative AI?"
        result = MarketMentor().crew().kickoff(inputs={"question": question}) 
    else:
        result = MarketMentor().crew().kickoff(inputs={"question": question})

    with open("MarketingStrategy.txt", "w", encoding="utf-8") as f:
        f.write(str(result))
    print("Final Result:", result)
    
if __name__ == "__main__":
    run()