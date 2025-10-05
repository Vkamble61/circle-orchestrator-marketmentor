import os
import yaml
from crewai import Agent, Task, Crew, Process, LLM
from crewai.project import CrewBase, agent, crew, task
#from crewai_tools import PDFSearchTool

from src.market_mentor.tools.extraction_tool import ExtractionTool

@CrewBase
class MarketMentor():
    def __init__(self):
        super().__init__()

        # Initialize the LLM with Ollama configuration
        self.llm = LLM(
            model=f"ollama/{os.getenv('OLLAMA_MODEL', 'llama3.2')}",
            base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
            api_key="NA"
            )
        
        # Load agent and task configurations from YAML files
        with open('src/market_mentor/config/agents.yaml', 'r', encoding='utf-8') as f:
            self.agents_config = yaml.safe_load(f)
        with open('src/market_mentor/config/tasks.yaml', 'r', encoding='utf-8') as f:
            self.tasks_config = yaml.safe_load(f)

        #self.question = "Brand Voice Consistency and Content Authenticity "
        
        self.extraction_tool = ExtractionTool() # Initialize the PDF extraction tool
        
    @agent
    def ingestion_agent(self) -> Agent:
        ingestion_cfg = self.agents_config['ingestion_agent']
        
        return Agent(
            role=ingestion_cfg['role'],
            goal=ingestion_cfg['goal'],
            backstory=ingestion_cfg['backstory'],
            verbose=True,
            llm=self.llm,                 
            max_iter=3
        )
    
    # @agent
    # def indexing_agent(self) -> Agent:
    #     indexing_cfg = self.agents_config['indexing_agent']

    #     return Agent(
    #         role=indexing_cfg['role'],
    #         goal=indexing_cfg['goal'],
    #         backstory=indexing_cfg['backstory'],
    #         verbose=True,
    #         llm=self.llm,
    #         max_iter=1
    #     )
    # @agent
    # def retriever_agent(self) -> Agent:
    #     retrieval_cfg = self.agents_config['retriever_agent']

    #     return Agent(
    #         role=retrieval_cfg['role'],
    #         goal=retrieval_cfg['goal'],
    #         backstory=retrieval_cfg['backstory'],
    #         verbose=True,
    #         llm=self.llm,
    #         max_iter=1
    #     )
    
    @task
    def ingestion_task(self) -> Task:
        if 'ingestion_task' not in self.tasks_config:
            raise KeyError("tasks_config does not contain 'ingestion_task'. Current keys: " + str(self.tasks_config.keys()))
        return Task(
            agent=self.ingestion_agent(),
            description=self.tasks_config['ingestion_task']['description'],
            expected_output=self.tasks_config['ingestion_task']['expected_output'],
            output_file="ingestion_output.txt",
            tools=[self.extraction_tool]
        )
    
    # @task
    # def indexing_task(self) -> Task:
    #     if 'indexing_task' not in self.tasks_config:
    #         raise KeyError("tasks_config does not contain 'indexing_task'. Current keys: " + str(self.tasks_config.keys()))
    #     return Task(
    #         agent=self.indexing_agent(),
    #         description=self.tasks_config['indexing_task']['description'],
    #         expected_output=self.tasks_config['indexing_task']['expected_output'],
    #         context=[self.ingestion_task()]

    #     )
    # @task
    # def retrieval_task(self) -> Task:
    #     if 'retrieval_task' not in self.tasks_config:
    #         raise KeyError("tasks_config does not contain 'retrieval_task'. Current keys: " + str(self.tasks_config.keys()))
    #     return Task(
    #         agent=self.retriever_agent(),
    #         description=self.tasks_config['retrieval_task']['description'] + " " + f"User question: {self.question}",
    #         expected_output=self.tasks_config['retrieval_task']['expected_output'],
            
    #     )
    

    @crew
    def crew(self ) -> Crew:
        #agents = [self.ingestion_agent(), self.indexing_agent(), self.retriever_agent()]
        #tasks = [self.ingestion_task(), self.indexing_task(), self.retrieval_task()] 
        agents = [self.ingestion_agent()]
        tasks = [self.ingestion_task()]
        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
            manager_llm=self.llm
        )
    
