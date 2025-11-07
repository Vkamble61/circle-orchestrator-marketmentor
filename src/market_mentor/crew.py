"""Market Mentor Crew Definition"""
import os
import yaml
from dotenv import load_dotenv

from crewai import Crew, Agent, Process, Task, LLM

from crewai.project import CrewBase, agent, task, crew
from crewai.memory.long_term.long_term_memory import LongTermMemory
from crewai.memory.short_term.short_term_memory import ShortTermMemory
from crewai.memory.entity.entity_memory import EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage 
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
try:
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import OllamaEmbeddings

from src.market_mentor.tools.extraction_tool import ExtractionTool
from src.market_mentor.tools.indexing_tool import IndexingTool
from src.market_mentor.tools.retrieval_tool import RetrievalTool

extraction_tool = ExtractionTool()
indexing_tool = IndexingTool()
retrieval_tool = RetrievalTool()

#load_dotenv()


@CrewBase
class MarketMentor():
    def __init__(self):
        # Load agent and task configurations from YAML files
        with open('src/market_mentor/config/agents.yaml', 'r', encoding='utf-8') as f:
            self.agents_config = yaml.safe_load(f)
        with open('src/market_mentor/config/tasks.yaml', 'r', encoding='utf-8') as f:
            self.tasks_config = yaml.safe_load(f)

        # Option 1: Keep local Ollama (slower but free)
        self.llm = LLM(
            model="ollama/llama3.2",
            base_url="http://localhost:11434",
            api_key=""
            #temperature=0.1,
            #max_tokens=1024,            
        )
        self.embedder = OllamaEmbeddings(
            model="nomic-embed-text",   
            base_url="http://localhost:11434"
        )
        # OpenAI       
        # self.llm = LLM(
        #     model="gpt-3.5-turbo",
        #     api_key=os.getenv("OPENAI_API_KEY"),
        #     temperature=0.1,
        #     max_tokens=1024
        # )     
        
    # @property
    # def agents(self):
    #     return [ self.ingestion_agent(), self.indexing_agent(), self.retrieval_agent(), self.marketing_agent()]
    
    # @property
    # def tasks(self):
    #     return [ self.ingestion_task(), self.indexing_task(), self.retrieval_task(), self.marketing_task()]
    
    @agent
    def ingestion_agent(self) -> Agent:
        ingest_config = self.agents_config.get('ingestion_agent', {})
        
        return Agent(
            role = ingest_config['role'],            
            goal = ingest_config['goal'],          
            backstory=ingest_config['backstory'],                   
            llm=self.llm,   
            max_iter=1,  # Absolutely stop after one iteration
            max_execution_time=180,  # Shorter timeout to prevent loops
            #allow_delegation=False,  # Don't delegate to other agents
            tools=[extraction_tool],
            verbose=True
        )
    
    @agent
    def indexing_agent(self) -> Agent:
        indexing_config = self.agents_config.get('indexing_agent', {})

        return Agent(
            role = indexing_config['role'],
            goal = indexing_config['goal'],
            backstory=indexing_config['backstory'],
            llm=self.llm,
            max_iter=1,  # Stop after first try
            max_execution_time=180,  # 3 minute timeout
            tools=[indexing_tool],
            verbose=True
        )

    @agent
    def retrieval_agent(self) -> Agent:
        retrieval_config = self.agents_config.get('retrieval_agent', {})
        
        return Agent(
            role = retrieval_config['role'],            
            goal = retrieval_config['goal'],        
            backstory=retrieval_config['backstory'],           
            llm=self.llm,   
            max_iter=1,  # Stop after first tool execution
            max_execution_time=180,  # 3 minute timeout
            tools=[retrieval_tool],
            verbose=True
        )
    
    @agent
    def marketing_agent(self) -> Agent:
        marketing_config = self.agents_config.get('marketing_agent', {})
        
        return Agent(
            role = marketing_config['role'],            
            goal = marketing_config['goal'],        
            backstory=marketing_config['backstory'],           
            llm=self.llm,   
            max_iter=1,
            max_execution_time=180,  # 3 minute timeout
            verbose=True
        )
   
    @task
    def ingestion_task(self) -> Task:
        ingest_config = self.tasks_config.get('ingestion_task', {})    
        
        return Task(
            agent=self.ingestion_agent(),
            description=ingest_config['description'],
            expected_output=ingest_config['expected_output']           
        )

    @task
    def indexing_task(self) -> Task:
        indexing_config = self.tasks_config.get('indexing_task', {})

        return Task(
            agent=self.indexing_agent(),
            description=indexing_config['description'],
            expected_output=indexing_config['expected_output'],
            #context=[self.ingestion_task()]  # Wait for ingestion to complete
        )
    
    @task
    def retrieval_task(self) -> Task:
        retrieval_config = self.tasks_config.get('retrieval_task', {})
        
        return Task(
            agent=self.retrieval_agent(),
            description=retrieval_config['description'],
            expected_output=retrieval_config['expected_output'],
            #context=[self.indexing_task()]  # Wait for indexing to complete
        )
    
    @task
    def marketing_task(self) -> Task:
        marketing_config = self.tasks_config.get('marketing_task', {})
    
        return Task(
            agent=self.marketing_agent(),
            description=marketing_config['description'],
            expected_output=marketing_config['expected_output'],
            context=[self.retrieval_task()]
        )
        
    
    @crew
    def crew(self) -> Crew:
        # Initialize long-term and short-term memory

        #storage_path = "./storage"

        # short_term_storage=RAGStorage(
        #         type="short_term",
        #         allow_reset=True,
        #         path=f"{storage_path}/chroma.sqlite3",
        #     )
        # short_term_memory_instance = ShortTermMemory(storage=short_term_storage)

        #Temporarily disable memory to avoid ChromaDB conflicts
        # ltm_storage = LTMSQLiteStorage(db_path=f"{storage_path}/chroma.sqlite3")
        # long_term_memory_instance = LongTermMemory(storage=ltm_storage, path=storage_path)
        
        # entity_memory_instance = EntityMemory(
        #     storage=RAGStorage(
        #         type="entity",
        #         allow_reset=True,
        #         path=f"{storage_path}/chroma.sqlite3"
        #     )
        # )

        agents = [self.ingestion_agent(), self.indexing_agent(), self.retrieval_agent(), self.marketing_agent()]  
        tasks = [self.ingestion_task(), self.indexing_task(), self.retrieval_task(), self.marketing_task()] 

        return Crew(
            agents=agents,
            tasks=tasks,            
            verbose=True,  # Enable detailed logging
            process=Process.sequential,

            # memory=True,
            # short_term_memory=short_term_memory_instance,
            # long_term_memory=long_term_memory_instance,
            # entity_memory=entity_memory_instance

            #max_rpm=10,  # Rate limit to prevent rapid iterations
            #step_callback=None,  # Disable callbacks that might cause loops
                       
            # planning=True,
            # planning_llm="llama3.2"
        )