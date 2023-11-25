# Databricks notebook source
# MAGIC %md
# MAGIC # Creating Serving Endpoints and Testing

# COMMAND ----------

# MAGIC %pip install -U mlflow==2.8.0 

# COMMAND ----------

from sentence_transformers import SentenceTransformer
import mlflow

# COMMAND ----------

username = spark.sql("SELECT current_user()").first()['current_user()']

model_name='sentence-transformers/all-mpnet-base-v2'

# UC Catalog Settings
use_uc = True
catalog = 'daiwt' # PLACE HERE YOUR CATALOG NAME 
db = 'llm' # PLACE HERE YOUR SCHEMA NAME 
uc_model_name = 'hf_embedding_model_bootcamp'

# mlflow settings
experiment_name = f'/Users/{username}/rag_llm_embedding'
run_name = 'embedding_model'
artifact_path = 'embedding_model'

# model serving settings
endpoint_name = 'hf_embedding_bootcamp_endpoint'
workload_sizing = 'Small'

# With GPU Private preview will have: workload_type
# {“CPU”, “GPU_MEDIUM”, “MULTIGPU_MEDIUM”} (AWS) 
# {“CPU”, “GPU_SMALL”, “GPU_LARGE”} (Azure)
workload_type = "CPU"

# COMMAND ----------

try:
  if use_uc:
    spark.sql(f'CREATE CATALOG IF NOT EXISTS {catalog}')
    spark.sql(f'CREATE SCHEMA IF NOT EXISTS {catalog}.{db}')

except:
  print("Seems like you do not have rigths to create a catalog, check if you have already one to be used or ask your administrator to set up one for")

#Enable Unity Catalog with mlflow registry
try:
  print('experiment exists already')
  mlflow.set_experiment(experiment_name)
  #mlflow.create_experiment(experiment_name)
except:
  print('experiment does not exist')
  mlflow.create_experiment(experiment_name)

# We need to know the Run id first. When running this straight then we can extract the run_id
if use_uc:
   mlflow.set_registry_uri('databricks-uc')
   register_name = f"{catalog}.{db}.{uc_model_name}"
else:
   register_name = uc_model_name

# COMMAND ----------

# DBTITLE 1,Setting Up the mlflow experiment

embedding_model = SentenceTransformer(model_name)

# Lets create a signature example
example_sentences = ["Welcome to sentence transformers", 
                    "This model is for embedding sentences"]

embedding_signature = mlflow.models.infer_signature(
    model_input=example_sentences,
    model_output=embedding_model.encode(example_sentences)
)

with mlflow.start_run(run_name=run_name) as run:
    mlflow.sentence_transformers.log_model(embedding_model,
                                  artifact_path=artifact_path,
                                  signature=embedding_signature,
                                  input_example=example_sentences)
    

# COMMAND ----------

# DBTITLE 1,Register Model
from mlflow import MlflowClient
client = mlflow.MlflowClient()

latest_model = mlflow.register_model(f'runs:/{run.info.run_id}/{artifact_path}', 
                                     register_name)

client.set_registered_model_alias(name=register_name, 
                                  alias="prod", 
                                  version=latest_model.version)

# COMMAND ----------



# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput

browser_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
db_host = f"https://{browser_host}"
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

w = WorkspaceClient(host=db_host, token=db_token)

# Specify the type of compute (CPU, GPU_SMALL, GPU_MEDIUM, etc.)
# Choose GPU_MEDIUM on AWS, and `GPU_LARGE` on Azure
#workload_type = "GPU_MEDIUM"
config = EndpointCoreConfigInput.from_dict({
    "served_models": [
        {
            "name": f'{latest_model.name.replace(".", "_")}_{latest_model.version}',
            "model_name": latest_model.name,
            "model_version": latest_model.version,
            "workload_type": workload_type,
            "workload_size": workload_sizing,
            "scale_to_zero_enabled": "False",
        }
    ]
})
w.serving_endpoints.create(name=endpoint_name, config=config)

# COMMAND ----------


