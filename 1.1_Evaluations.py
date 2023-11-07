# Databricks notebook source
# MAGIC %md
# MAGIC # Evaluations
# MAGIC Running Evaluations on RAGs is still more art than science \
# MAGIC We will use llama_index to assist in generating evaluation questions \
# MAGIC And use the inbuilt assessment prompt in llama_index \

# COMMAND ----------

# MAGIC %pip install llama_index==0.8.54 spacy ragas==0.0.18 mlflow==2.8.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import nest_asyncio
# Needed for the async calls to work
nest_asyncio.apply()

# COMMAND ----------

# MAGIC %md
# MAGIC # Intro to Llama Index
# MAGIC Much like langchain, llama_index is an orchestration layer for LLM logic \
# MAGIC Where they differ is that llama_index is a lot more focused on RAGs and doing intelligent indexing \
# MAGIC Langchain is more generalist and has been focused on enabling complex workflows
# MAGIC
# MAGIC Llama Index has a few key concepts we will use for this notebook:
# MAGIC - Service Context - wrapper class to hold llm model / embeddings
# MAGIC - An Index - this is the core of llama index. At it's base, an index consists of a complex structure of nodes which contain text and embeddings 

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

#import os

# COMMAND ----------

# DBTITLE 1,Configurations
test_pdf = '/dbfs/bootcamp_data/pdf_data/2302.09419.pdf'
#test_pdf = f'{dbfs_source_docs}/2302.09419.pdf'
test_pdf

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Service Context
# MAGIC The service context sets up the LLM and embedding model that we will use for our exercises
# MAGIC In this case, the Embedding Model and the LLM are both setup onto Databricks serving

# COMMAND ----------

from llama_index import (
  ServiceContext,
  set_global_service_context,
  LLMPredictor
)
from llama_index.callbacks import CallbackManager, LlamaDebugHandler, CBEventType

# Using Databricks Model Serving
browser_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
db_host = f"https://{browser_host}"
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

serving_uri = 'hf_inference_bootcamp_endpoint'
serving_model_uri = f"{db_host}/serving-endpoints/{serving_uri}/invocations"

embedding_uri = 'hf_embedding_bootcamp_endpoint'
embedding_model_uri = f"{db_host}/serving-endpoints/{embedding_uri}/invocations"


# COMMAND ----------

# Embedding wrapper

from typing import Any, Dict, List, Mapping, Optional, Tuple

import requests

from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.schema.embeddings import Embeddings
from langchain.utils import get_from_dict_or_env


class ModelServingEndpointEmbeddings(BaseModel, Embeddings):
    """Databricks Model Serving embedding service.

    To use, you should have the
    environment variable ``DB_API_TOKEN`` set with your API token, or pass
    it as a named parameter to the constructor.
    """

    endpoint_url: str = (
        "https://e2-dogfood.staging.cloud.databricks.com/serving-endpoints/hf_embedding_bootcamp_endpoint/invocations"
    )
    """Endpoint URL to use."""
    embed_instruction: str = "Represent the document for retrieval: "
    """Instruction used to embed documents."""
    query_instruction: str = (
        "Represent the question for retrieving supporting documents: "
    )
    """Instruction used to embed the query."""
    retry_sleep: float = 1.0
    """How long to try sleeping for if a rate limit is encountered"""

    db_api_token: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        db_api_token = get_from_dict_or_env(
            values, "db_api_token", "DB_API_TOKEN"
        )
        values["db_api_token"] = db_api_token
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"endpoint_url": self.endpoint_url}

    def _embed(
        self, input: List[Tuple[str, str]], is_retry: bool = False
    ) -> List[List[float]]:
        #payload = {"input_strings": input}
        payload = {
            "dataframe_split": {
                "data": [
                    [
                        input
                    ]
                ]
            }
        }

        # HTTP headers for authorization
        headers = {
            "Authorization": f"Bearer {self.db_api_token}",
            "Content-Type": "application/json",
        }

        # send request
        try:
            response = requests.post(url=self.endpoint_url, headers=headers, json=payload)
        
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        try:
            if response.status_code == 429:
                if not is_retry:
                    import time
                    time.sleep(self.retry_sleep)
                    return self._embed(input, is_retry=True)
                raise ValueError(
                    f"Error raised by inference API: rate limit exceeded.\nResponse: "
                    f"{response.text}"
                )
            
            parsed_response = response.json()

        except requests.exceptions.JSONDecodeError as e:
            raise ValueError(
                f"Error raised by inference API: {e}.\nResponse: {response.text}"
            )
        return parsed_response

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using a MosaicML deployed instructor embedding model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        try:
            embeddings = [self._embed(x)['predictions'][0] for x in texts]
        except KeyError:
            print([self._embed(x) for x in texts])

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using a Databricks Model Serving embedding model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embedding = self._embed(text)
        return embedding['predictions']

# COMMAND ----------


llm_model = ServingEndpointLLM(endpoint_url=serving_model_uri, token=db_token)

llm_predictor = LLMPredictor(llm=llm_model)

### define embedding model setup
embeddings = ModelServingEndpointEmbeddings(db_api_token=db_token)

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, 
                                               embed_model=embeddings,
                                               callback_manager = callback_manager 
                                               )

# we can now set this context to be a global default
set_global_service_context(service_context)

# COMMAND ----------

# MAGIC %md
# MAGIC # Load and Chunk Document
# MAGIC We will load a sample doc to test on, firstly with a naive default chunking strategy

# COMMAND ----------

# chunk the output
from llama_index import (
    download_loader, VectorStoreIndex
)
from pathlib import Path

PDFReader = download_loader('PDFReader')
loader = PDFReader()

# This produces a list of llama_index document objects
documents = loader.load_data(file=Path(test_pdf))

# COMMAND ----------

documents

# COMMAND ----------

# we are just setting up a simple in memory Vectorstore here
index = VectorStoreIndex.from_documents(documents)

# and turning it into a query engine
query_engine = index.as_query_engine()

# Let's validate that it is all working
reply = query_engine.query('what is a neural network?')

# COMMAND ----------

print(reply.response)

# COMMAND ----------

# MAGIC %md
# MAGIC # Build out evaluation Questions
# MAGIC In order to run evaluation we need to have feasible questions to feed the model \
# MAGIC It is time consuming to manually construct questions so we will use a LLM to do this \
# MAGIC Note that this will have limitations, namely in the types of questions it will generate

# COMMAND ----------

from llama_index.evaluation import DatasetGenerator, RelevancyEvaluator

# this is the question generator. Note that it has additional settings to customise prompt etc
data_generator = DatasetGenerator.from_documents(documents=documents, service_context=service_context)

# this is the call to generate the questions
eval_questions = data_generator.generate_questions_from_nodes()

# Some of these questions might not be too useful. It could be because of the model we are using for generation
# It could also be that the chunk is particularly bad

# COMMAND ----------

len(eval_questions)

# COMMAND ----------

eval_questions[:20]

# COMMAND ----------

# MAGIC %md
# MAGIC # Use Questions to generate evaluations
# MAGIC Now we have our queries we need to run some responses
# MAGIC
# MAGIC This next step can be slow so we will cut it down to 20 questions \
# MAGIC We can then use the `ResponseEvaluator`` looks at whether the query is answered by the response

# COMMAND ----------

import pandas as pd

eval_questions = eval_questions[0:20]

# Yes we are using a LLM to evaluate a LLM
## When doing this normally you might use a more powerful but more expensive evaluator
## to assess the quality of your input
evaluator = RelevancyEvaluator(service_context=service_context)

# lets create and log the data properly
results = []

for question in eval_questions:
    
    engine_response = query_engine.query(question)
    evaluation = evaluator.evaluate_response(question, engine_response)
    results.append(
      {
        "query": question,
        "response": str(engine_response.response),
        "source": engine_response.source_nodes[0].node.text,
        "evaluation": evaluation
      }   
    )

# we will load it into a pandas frame: 
response_df = pd.DataFrame(results)

# COMMAND ----------

# Let see what is in the frame
response_df

# COMMAND ----------


