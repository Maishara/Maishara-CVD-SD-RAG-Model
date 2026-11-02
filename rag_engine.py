import os
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from schemas import SystemDynamicsParameterSchema, RAGExtractionResponse

class ClinicalRAGManager:
    """
    RAG MAnagement Engine using LangChain, Chroma vector store, and structured Pydantic parameter etraction for System Dynamics modeling.
    Grounded in 2010 UK health baseline (BHF, NICE NG238, NHS QOF)
    """
    def __init__(self, persist_directory: str = "/chorma_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = Chroma(
           collection_name="uk_clinical_benchmarks_2010",
           embedding_function=self.embeddings,
              persist_directory=self.persist_directory
        )
        # Keeps the model that works on your account tier
        self.llm = ChatAnthropic(
            model="claude-sonnet-5-5"
        )

    def populate_mock_knowledge_base(self):
        """Populates vector store with sample benchmark records from BHF/NHS."""
        clinical_documents = [
            "BHF Statistics Compendium 2025: Total CVD prevalence in UK adults aged 40-70 is estimated at 6.8% (0.068). Annual new incidence rate is 0.8% (0.008) per year.",
            "BHF Factsheet 2024: Cigarette smoking increases cardiovascular disease incidence with an average risk multiplier of 1.50 compared to non-smokers.",
            "NHS Digital QOF Guidance: Diagnosed hypertension (high blood pressure) carries a relative risk multiplier of 1.40 for major adverse cardiovascular events.",
            "NICE CVD Prevention Guidelines (NG238): Community smoking cessation programs demonstrate a 15% (0.15) reduction in 10-year CVD incidence rates."
        ]
        self.vector_store.add_texts(
            texts=clinical_documents,
            metadatas=[{"source": "BHF 2025"}, {"source": "BHF 2024"}, {"source": "NHS QOF"}, {"source": "NICE NG238"}]
        )

    def retrieve_clinical_context(self, query: str, k: int = 3) -> List[str]:
        """Performs semantic search across indexed clinical documents."""
        results = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

    def extract_calibrated_parameters(self, user_scenario_prompt: str) -> RAGExtractionResponse:
        """
        Combines semantic retrieval with Pydantic structured output generation.
        """
        # 1. Retrieve Context
        retrieved_chunks = self.retrieve_clinical_context(user_scenario_prompt)
        context_str = "\n".join([f"- {chunk}" for chunk in retrieved_chunks])

        # 2. Use json_mode to bypass forced tool-calling parameter overrides
        structured_llm = self.llm.with_structured_output(
            SystemDynamicsParameterSchema, 
            method="json_mode"
        )

        system_prompt = f"""
        You are an expert epidemiological parameter extraction system.
        Extract and calibrate System Dynamics model parameters using ONLY the provided clinical context below.
        
        CLINICAL CONTEXT:
        {context_str}
        
        Ensure all extracted numeric parameters fit realistic bounds and match the user scenario.
        Respond STRICTLY with a valid JSON object matching the requested schema.
        """

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input_scenario}")
        ])

        chain = prompt_template | structured_llm
        calibrated_params = chain.invoke({"input_scenario": user_scenario_prompt})

        return RAGExtractionResponse(
            retrieved_context_chunks=retrieved_chunks,
            parameters=calibrated_params
        )