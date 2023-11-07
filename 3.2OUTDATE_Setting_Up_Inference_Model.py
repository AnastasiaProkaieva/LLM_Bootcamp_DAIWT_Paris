# Databricks notebook source
# MAGIC %md
# MAGIC # Creating Serving Endpoints and Testing

# COMMAND ----------

# MAGIC %pip install -U bitsandbytes==0.40.0 transformers==4.31.0 mlflow==2.8.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from transformers import (
   AutoModelForCausalLM,
   AutoTokenizer,
   AutoConfig,
   pipeline
)
import mlflow
import torch

# COMMAND ----------

import huggingface_hub
huggingface_key = dbutils.secrets.get(scope='bootcamp_paris', key='HF_TOKEN')
huggingface_hub.login(token=huggingface_key)

# COMMAND ----------

username = spark.sql("SELECT current_user()").first()['current_user()']

model_name = 'meta-llama/Llama-2-7b-chat-hf'
revision = '08751db2aca9bf2f7f80d2e516117a53d7450235'

# UC Catalog Settings
use_uc = True
catalog = 'ap'
db = 'llm'
uc_model_name = 'hf_inference_model_bootcamp'

# mlflow settings
experiment_name = f'/Users/{username}/rag_llm_inference'
run_name = 'inference_model'
artifact_path = 'inference_model'

# model serving settings
endpoint_name = 'hf_inference_bootcamp_endpoint'
workload_sizing = 'Small'

# With GPU Private preview will have: workload_type
# {“CPU”, “GPU_MEDIUM”, “MULTIGPU_MEDIUM”} (AWS) 
# {“CPU”, “GPU_SMALL”, “GPU_LARGE”} (Azure)
workload_type = "GPU_MEDIUM"


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

# DBTITLE 1,Setting Up a Model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_config = AutoConfig.from_pretrained(
    model_name,
    trust_remote_code=True,  # this can be needed if we reload from cache
    revision=revision,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    revision=revision,
    trust_remote_code=True,  # this can be needed if we reload from cache
    config=model_config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="cuda",
    # load_in_8bit=True
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

inference_config = {"do_sample": True, "max_new_tokens": 512}


# Lets create a signature example
example_sentences = [
    "<s>[INST]<<SYS>>Answer questions succintly<</SYS>> Who are you?[/INST]",
    "<s>[INST]<<SYS>>Answer questions succintly<</SYS>> How can you help me?[/INST]",
]

# COMMAND ----------

# DBTITLE 1,Setting Up the mlflow experiment


#LLama 2 special type currently not supported
embedding_signature = mlflow.models.infer_signature(
    model_input=example_sentences[0],
    model_output=pipe(example_sentences[0])
)

with mlflow.start_run(run_name=run_name) as run:
    mlflow.transformers.log_model(pipe,
                                  artifact_path=artifact_path,
                                  signature=embedding_signature,
                                  input_example=example_sentences,
                                  inference_config=inference_config,
                                  pip_requirements={
                                    'transformers==4.31.0'}
                                  )
    

# COMMAND ----------

# DBTITLE 1,Register Model

client = mlflow.MlflowClient()

# We need to know the Run id first. When running this straight then we can extract the run_id
latest_model = mlflow.register_model(f'runs:/{run.info.run_id}/{artifact_path}', 
                                     f"{catalog}.{db}.{uc_model_name}")

client.set_registered_model_alias(name=f"{catalog}.{db}.{uc_model_name}", 
                                  alias="prod", 
                                  version=latest_model.version)

# COMMAND ----------

# DBTITLE 1,Deploy Endpoint
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
            "workload_type": "GPU_MEDIUM",
            "workload_size": workload_sizing,
            "scale_to_zero_enabled": "False",
        }
    ]
})
w.serving_endpoints.create(name=endpoint_name, config=config)

# COMMAND ----------

f'{latest_model.name.replace(".", "_")}_{latest_model.version}'

# COMMAND ----------


