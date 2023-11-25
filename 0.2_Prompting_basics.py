# Databricks notebook source
# MAGIC %md
# MAGIC # Prompting Basics
# MAGIC
# MAGIC Lets explore the basics of prompting\
# MAGIC For more details see: https://www.promptingguide.ai/

# COMMAND ----------

# DBTITLE 1,Library Setup
#%pip install ctransformers==0.2.26
%pip install mlflow==2.8.0 llama_index==0.8.54

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

run_mode = 'serving' # or gpu or cpu or serving

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %run ./utils

# COMMAND ----------

# MAGIC %md 
# MAGIC Payload example to paly with the UI 
# MAGIC
# MAGIC ```
# MAGIC
# MAGIC {
# MAGIC   "dataframe_split": {
# MAGIC     "columns": [
# MAGIC       "prompt",
# MAGIC       "temperature",
# MAGIC       "max_new_tokens"
# MAGIC     ],
# MAGIC     "data": [
# MAGIC       [
# MAGIC         "what is ML?",
# MAGIC         0.5,
# MAGIC         100
# MAGIC       ]
# MAGIC     ]
# MAGIC   }
# MAGIC }
# MAGIC
# MAGIC ```

# COMMAND ----------

pipe = load_model(run_mode, dbfs_tmp_cache) #, 'zephyr_7b'

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Prompting Techniques

# COMMAND ----------

# MAGIC %md
# MAGIC # Basic Prompting
# MAGIC Getting started is easy, we can send text in.
# MAGIC Remember that different models will respond differently!
# MAGIC The same model can also responds differently when we rerun a prompt
# MAGIC (though you likely only see this with basic one line prompts)

# COMMAND ----------

prompt = "The sky is"
select_proper_set(prompt, run_mode)

# COMMAND ----------

prompt = """
    Knock Knock
    Who's there?
    """
select_proper_set(prompt, run_mode, max_new_tokens=250, max_tokens=100)

# COMMAND ----------

# MAGIC %md
# MAGIC # Zero Shot Prompting
# MAGIC Zero shot is the most basic way to ask something of the model.
# MAGIC Just define your task and ask!

# COMMAND ----------

prompt = """
    Classify the text into neutral, negative or positive.
    Text: I think the vacation is okay.
    Sentiment:
"""

select_proper_set(prompt, run_mode, max_new_tokens=250, max_tokens=100)

# COMMAND ----------

# MAGIC %md
# MAGIC You might have gotten some rubbish, we did the first time. (note models are stochastic) 
# MAGIC And that is because our prompt is problematic.
# MAGIC Different models have different "prompt templates" that they use.
# MAGIC Let's try using the official one for Llama 2

# COMMAND ----------

prompt = """<s>[INST]<<SYS>>Classify the text into neutral, negative or positive.<</SYS>>

Text: I think the vacation is okay.
Sentiment: [/INST]
"""

zephyr_prompt = """<|system|>
Classify the text into neutral, negative or positive.

<|user|>
Text: I think the vacation is okay.
Sentiment: 
<|assistant|> 
"""

# COMMAND ----------

select_proper_set(prompt, run_mode, max_new_tokens=100, max_tokens=100)

# COMMAND ----------

select_proper_set(zephyr_prompt, run_mode, max_new_tokens=100, max_tokens=100)

# COMMAND ----------

# MAGIC %md llama 2 uses the [INST] tag to highlight the whole instruction
# MAGIC <<SYS>> is the system prompt, the guide for the model on how to respond
# MAGIC In our sample the user question comes after the Text: field
# MAGIC you should get a better response after adopting this format.

# COMMAND ----------

prompt = """<s>[INST]<<SYS>>Provide an answer to the question based on the following:<</SYS>>

The minutes from the Fed's June 13-14 meeting show that while almost all officials deemed it “appropriate or acceptable” to keep rates unchanged in a 5% to 5.25% target range, some would have supported a quarter-point increase instead.

User Question: What is the interest rate in following paragraph?
Answer: [/INST]
"""

zephyr_prompt = """<|assistant|>
Provide an answer to the question based on the following:

The minutes from the Fed's June 13-14 meeting show that while almost all officials deemed it “appropriate or acceptable” to keep rates unchanged in a 5% to 5.25% target range, some would have supported a quarter-point increase instead.

<|user|>
User Question: What is the interest rate in following paragraph?
Answer:
<|assistant|>
"""


# COMMAND ----------

select_proper_set(prompt, run_mode, max_new_tokens=100, max_tokens=100)

# COMMAND ----------

select_proper_set(zephyr_prompt, run_mode, max_new_tokens=100, max_tokens=100)

# COMMAND ----------

# MAGIC %md
# MAGIC # Few Shot Prompting
# MAGIC One way to help the a model do logic better is to provide it with samples
# MAGIC

# COMMAND ----------

prompt = """
<s>[INST]<<SYS>>
Be helpful and suggest a type of account for a customer.
<</SYS>>    

Here are some examples:
A consumer wants a savings account
A business wants a business account
A tech unicorn deserves a special VC account

Question:
What account would you recommend a small business?[/INST]
"""

zephyr_prompt = """<|system|>
Be helpful and suggest a type of account for a customer.

Here are some examples:
A consumer wants a savings account
A business wants a business account
A tech unicorn deserves a special VC account

Question:
<|user|>
What account would you recommend a small business?[/INST]

<|assistant|>
"""


# COMMAND ----------

select_proper_set(prompt, run_mode, max_new_tokens=100, max_tokens=100)

# COMMAND ----------

select_proper_set(zephyr_prompt, run_mode, max_new_tokens=100, max_tokens=100)

# COMMAND ----------

prompt = """
<s>[INST]<<SYS>>
Be helpful and suggest a type of account for a customer.
<</SYS>>    

Here are some examples:
A consumer wants a savings account
A business wants a business account
A tech unicorn deserves a special VC account

Question:
What account would you recommend a bob the builder?[/INST]
"""

zephyr_prompt = """<|system|>
Be helpful and suggest a type of account for a customer.

Here are some examples:
A consumer wants a savings account
A business wants a business account
A tech unicorn deserves a special VC account

Question:
<|user|>
What account would you recommend a bob the builder?

<|assistant|>
"""


# COMMAND ----------

select_proper_set(prompt, run_mode, max_new_tokens=100, max_tokens=100)

# COMMAND ----------

select_proper_set(zephyr_prompt, run_mode, max_new_tokens=100, max_tokens=100)

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC # Chain of Thought Prompting
# MAGIC In chain of thought prompting, we show the model how to rationalise\
# MAGIC This can help it deduce how to do a task properly

# COMMAND ----------

# DBTITLE 1,Standard Prompt - we ask straight away
user_question = """The cafeteria had 23 apples. If they used 20 to
make lunch and bought 6 more, how many apples
do they have?"""

prompt = f"""
<s>[INST]<<SYS>>
Provide helpful responses and guide the customers. 
<</SYS>>    

The follow example shows how to answer:
Question:
I went to the market and bought 10 apples. 
I gave 2 apples to the neighbor and 2 to the repairman. 
I then went and bought 5 more apples and ate 1. 

Answer:
The answer is 10

Based on the above provide the answer to the following question.
Question:
{user_question}[/INST]
"""

zephyr_prompt = f"""
<|system|>
Provide helpful responses and guide the customers.     

The follow example shows how to answer:
Question:
I went to the market and bought 10 apples. 
I gave 2 apples to the neighbor and 2 to the repairman. 
I then went and bought 5 more apples and ate 1. 

Answer:
The answer is 10

Based on the above provide the answer to the following question.
<|user|>
{user_question}
<|assistant|>
"""

# COMMAND ----------

select_proper_set(prompt, run_mode, max_new_tokens=250, max_tokens=100)

# COMMAND ----------

select_proper_set(zephyr_prompt, run_mode, max_new_tokens=250, max_tokens=100)

# COMMAND ----------

# DBTITLE 1,Chain of Thought Prompt
user_question = """The cafeteria had 23 apples. If they used 20 to
make lunch and bought 6 more, how many apples
do they have?"""

prompt = f"""
<s>[INST]<<SYS>>
Provide helpful responses and guide the customers. 
<</SYS>>    

The follow example shows how to answer:
Question:
I went to the market and bought 10 apples. 
I gave 2 apples to the neighbor and 2 to the repairman. 
I then went and bought 5 more apples and ate 1. 

Answer:
We had 10 applies. We gave away 2 each to the neighbour and the repairman.
10 - 2 - 2 = 6. We bought 5 and ate 1. 6+5-1=10 The answer is 10

Based on the above provide the answer to the following question.
Question:
{user_question}[/INST]
"""


# COMMAND ----------

select_proper_set(prompt, run_mode, max_new_tokens=250, max_tokens=100)

# COMMAND ----------

select_proper_set(zephyr_prompt, run_mode, max_new_tokens=250, max_tokens=100)

# COMMAND ----------

# MAGIC %md
# MAGIC # System Prompts
# MAGIC Systems prompts can be used to instruct a model and also to tune it's reponse
# MAGIC You have already seen them. It is the bit inside the <<SYS>> tags.
# MAGIC They can have a big effect!
# MAGIC

# COMMAND ----------

system_prompt = 'Be helpful and suggest a type of account for a customer. try to be curteous and explain some of the key things to consider in bank account selection.'

user_question = 'I am a single homeless bloke what account should I get?'

prompt = f"""
<s>[INST]<<SYS>>
{system_prompt}
<</SYS>>    

Here are some examples:
A consumer wants a savings account
A business wants a business account
A tech unicorn deserves a special VC account

Question:
{user_question}[/INST]
"""

zephyr_prompt  = f"""
<|system|>
{system_prompt}
    
Here are some examples:
A consumer wants a savings account
A business wants a business account
A tech unicorn deserves a special VC account

<|user|>
{user_question}

<|assistant|>
"""

# COMMAND ----------

select_proper_set(prompt, run_mode, max_new_tokens=250, max_tokens=100)

# COMMAND ----------

select_proper_set(zephyr_prompt, run_mode, max_new_tokens=250, max_tokens=100)

# COMMAND ----------

system_prompt = 'As a learned English Gentleman, be helpful and suggest a type of account for a customer. try to be curteous and explain in flowery language some of the key things to consider in bank account selection.'

user_question = 'I am a single and jobless bloke what account suggest me a type of bank account?'

prompt = f"""
<s>[INST]<<SYS>>
{system_prompt}
<</SYS>>    

Question:
{user_question}[/INST]
"""

zehpyr_prompt = f"""<|system|>
{system_prompt}

<|user|>
{user_question}
<|assistant|>
"""


# COMMAND ----------

select_proper_set(prompt, run_mode, max_new_tokens=500, max_tokens=100)

# COMMAND ----------

select_proper_set(zephyr_prompt, run_mode, max_new_tokens=500, max_tokens=100)

# COMMAND ----------

# MAGIC %md
# MAGIC # Prompt Formatting
# MAGIC Prompt formats help to structure the prompts for different LLMs
# MAGIC Each LLM could have a different standard
# MAGIC
# MAGIC Stanford Alpaca structure
# MAGIC
# MAGIC ```
# MAGIC Below is an instruction that describes a task.
# MAGIC Write a response that appropriately completes the request.
# MAGIC ### Instruction:
# MAGIC {user question}
# MAGIC ### Response:
# MAGIC ```
# MAGIC
# MAGIC llama v2 format
# MAGIC ```
# MAGIC <s>[INST] <<SYS>>
# MAGIC You are a friendly assistant. Be Polite and concise.
# MAGIC <</SYS>>
# MAGIC
# MAGIC Answer the following question:
# MAGIC {user question}
# MAGIC [/INST]
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC # Retrieval Augmented Generation
# MAGIC
# MAGIC Now if I ask the bot about something left of field
# MAGIC It probably cannot answer
# MAGIC Training is expensive
# MAGIC What if we gave it an except?
# MAGIC
# MAGIC

# COMMAND ----------

system_prompt = 'As a helpful long island librarian, answer the questions provided in a succint and eloquent way.'

user_question = 'Explain to me like I am 5 LK-99'

prompt = f"""
<s>[INST]<<SYS>>
{system_prompt}
<</SYS>>    

Question:
{user_question}[/INST]
"""

zephyr_prompt = f"""<|system|>
{system_prompt}

<|user|>
{user_question}
<|assistant|>
"""


# COMMAND ----------

select_proper_set(prompt, run_mode, max_new_tokens=250, max_tokens=100)

# COMMAND ----------

select_proper_set(zephyr_prompt, run_mode, max_new_tokens=250, max_tokens=100)

# COMMAND ----------

system_prompt = 'As a helpful long island librarian, answer the questions provided in a succint and eloquent way.'

user_question = 'Explain to me like I am 5 LK-99'

prompt = f"""<s>[INST]<<SYS>>{system_prompt}<</SYS>>

Based on the below context:

LK-99 is a potential room-temperature superconductor with a gray‒black appearance.[2]: 8  It has a hexagonal structure slightly modified from lead‒apatite, by introducing small amounts of copper. A room-temperature superconductor is a material that is capable of exhibiting superconductivity at operating temperatures above 0 °C (273 K; 32 °F), that is, temperatures that can be reached and easily maintained in an everyday environment.

Provide an answer to the following:
{user_question}[/INST]
"""

zephyr_prompt = f"""<|system|>{system_prompt}

Based on the below context:

LK-99 is a potential room-temperature superconductor with a gray‒black appearance.[2]: 8  It has a hexagonal structure slightly modified from lead‒apatite, by introducing small amounts of copper. A room-temperature superconductor is a material that is capable of exhibiting superconductivity at operating temperatures above 0 °C (273 K; 32 °F), that is, temperatures that can be reached and easily maintained in an everyday environment.

Provide an answer to the following:
<|user|>
{user_question}

<|assistant|>
"""


# COMMAND ----------

select_proper_set(prompt, run_mode, max_new_tokens=250, max_tokens=100)

# COMMAND ----------

select_proper_set(zephyr_prompt, run_mode, max_new_tokens=250, max_tokens=100)

# COMMAND ----------


