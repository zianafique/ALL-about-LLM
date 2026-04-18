#!/usr/bin/env python
# coding: utf-8

# # LangChain: Evaluation
# 
# ## Outline:
# 
# * Example generation
# * Manual evaluation (and debuging)
# * LLM-assisted evaluation
# * LangChain evaluation platform

# In[1]:


import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


# Note: LLM's do not always produce the same results. When executing the code in your notebook, you may get slightly different answers that those in the video.

# In[2]:


# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"


# ## Create our QandA application

# In[42]:


from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch


# In[43]:


file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
data = loader.load()


# In[44]:


index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])


# In[45]:


llm = ChatOpenAI(temperature = 0.0, model=llm_model)
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=index.vectorstore.as_retriever(), 
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)


# ### Coming up with test datapoints

# In[46]:


data[2]


# In[47]:


data[25]


# ### Hard-coded examples

# In[48]:


examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set\
        have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty \
        850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    }
]


# ### LLM-Generated examples

# In[49]:


from langchain.evaluation.qa import QAGenerateChain


# In[50]:


example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI(model=llm_model))


# In[13]:


# the warning below can be safely ignored


# In[51]:


new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:10]]
)


# In[54]:


new_examples[9]


# In[55]:


data[5]


# ### Combine examples

# In[56]:


examples += new_examples


# In[57]:


qa.run(examples[5]["answer"])


# ## Manual Evaluation

# In[58]:


import langchain
langchain.debug = True


# In[59]:


qa.run(examples[10]["query"])


# In[60]:


# Turn off the debug mode
langchain.debug = False


# ## LLM assisted evaluation

# In[61]:


predictions = qa.apply(examples)


# In[62]:


from langchain.evaluation.qa import QAEvalChain


# In[63]:


llm = ChatOpenAI(temperature=0, model=llm_model)
eval_chain = QAEvalChain.from_llm(llm)


# In[64]:


graded_outputs = eval_chain.evaluate(examples, predictions)


# In[66]:


for i, eg in enumerate(examples):
    print(f"Example {i+1}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['text'])
    print()


# In[67]:


graded_outputs[0]

