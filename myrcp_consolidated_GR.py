# Databricks notebook source
!pip install langchain
!pip install openai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import openai
import json 
import pandas as pd
import numpy as np
from pyspark.sql.functions import lit, rand, col,current_timestamp
from pyspark.sql.types import *
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.llms import LlamaCpp
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers.combining import CombiningOutputParser
from langchain.llms import AzureOpenAI
from langchain.prompts import PromptTemplate 
from langchain.chains import LLMChain
from datetime import datetime, date, time, timedelta
import pandas as pd


openai.api_key =  dbutils.secrets.get(scope='myrcpscope1', key='myrcp-oai-primary-key')
openai.api_base = "https://oai-adp1reg-myrcp-dv-we-01.openai.azure.com/"
openai.api_type = 'azure'
openai.api_version = '2024-02-01' # this might change in the future
deployment_name='myrcp_dev' #This will correspond to the custom name you chose for your deployment when you deployed a model.

# Date to Process
dbutils.widgets.text("DateToProcess","2024-05-14")
DateToProcess = datetime.strptime(dbutils.widgets.get("DateToProcess"), '%Y-%m-%d')
DateToProcess_path = DateToProcess.strftime('%Y-%m-%d')
print(DateToProcess_path)
DateToProcess_filename = DateToProcess.strftime('%Y%m%d')
print(DateToProcess_filename)

#opco
dbutils.widgets.text("opco","RS")
opco = dbutils.widgets.get("opco")

# COMMAND ----------

# DBTITLE 1,ADLS Connect
storage_account_name = dbutils.secrets.get(scope='myrcpscope1', key='sa-name')
container_name = dbutils.secrets.get(scope='myrcpscope1', key='co-name')
storage_account_access_key = dbutils.secrets.get(scope='myrcpscope1', key='sa-accesskey')
application_id =  dbutils.secrets.get(scope='myrcpscope1', key='spnAppId-spn-becse-dv-dia-adp1reg-myrcp-01')
directory_id = dbutils.secrets.get(scope='myrcpscope1', key='tenant-id')
service_credential = dbutils.secrets.get(scope='myrcpscope1', key='spnSecret-spn-becse-dv-dia-adp1reg-myrcp-01')

spark.conf.set(
    f"fs.azure.account.auth.type.{storage_account_name}.dfs.core.windows.net",
    "OAuth",
)
spark.conf.set(
    f"fs.azure.account.oauth.provider.type.{storage_account_name}.dfs.core.windows.net",
    "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
)
spark.conf.set(
    f"fs.azure.account.oauth2.client.id.{storage_account_name}.dfs.core.windows.net",
    application_id,
)
spark.conf.set(
    f"fs.azure.account.oauth2.client.secret.{storage_account_name}.dfs.core.windows.net",
    service_credential,
)
spark.conf.set(
    f"fs.azure.account.oauth2.client.endpoint.{storage_account_name}.dfs.core.windows.net",
    f"https://login.microsoftonline.com/{directory_id}/oauth2/token",
)

# COMMAND ----------

root_path = f"abfss://myrcp@{storage_account_name}.dfs.core.windows.net/"
root_files = dbutils.fs.ls(root_path)
# display(root_files)

PO_path = f"{root_path}{opco}/PersonalOffers/{DateToProcess_path}/offer_allocations_{opco}_{DateToProcess_filename}.csv"
print(PO_path)
Article_Path = f"{root_path}{opco}/Article/DELTA_Identity/"
print(Article_Path)
ArticleHierarchy_Path = f"{root_path}{opco}/ArticleHierarchy/DELTA_Identity/"
print(ArticleHierarchy_Path)

# COMMAND ----------

from pyspark.sql import Row

#These are the article hierarchy that are food
food_art_hie = ['GS01', 'GS02' , 'GS03', 'GS04', 'GS05' , 'GS06','GS08','GS09','GS10' , 'GS11' ,'GS12' ,'GS13' , 'GS14', 'GS15','GS16', 'GS17','GS41','GS44','GS45','GS46']

# Create DataFrame from list
df = spark.createDataFrame(food_art_hie, "string")
df.createOrReplaceTempView("food_art_hie")

# COMMAND ----------

# DBTITLE 1,Read the Files from ADLS
POfferDf = spark.read.csv(f"{PO_path}",header=True)
POfferDf.createOrReplaceTempView('POffer')

ArticleDf = spark.read.format("delta").load(Article_Path)
ArticleDf.createOrReplaceTempView('Article')

ArticleHierDf = spark.read.format("delta").load(ArticleHierarchy_Path)
ArticleHierDf.createOrReplaceTempView('ArticleHier')

# COMMAND ----------

# DBTITLE 1,Filtering 100 customers
# Select distinct loyalty_card_hash
distinct_loyalty_card_hashes = POfferDf.select("customer_public_hash").distinct()

# Get X distinct loyalty_card_hash
ten_distinct_loyalty_card_hashes = distinct_loyalty_card_hashes.limit(20) #we can change here to test more customers or just remove the limit so we run all customers

# Convert to list
list_of_hashes = [row['customer_public_hash'] for row in ten_distinct_loyalty_card_hashes.collect()]

print(list_of_hashes)

# COMMAND ----------

# DBTITLE 1,SQL - Main FIlter Query and Convert Col to List
main_list = []  # Create an empty main list

for hashes in list_of_hashes:
  query = f'''SELECT  a.promotion_header_id,
  a.customer_public_hash ,
  a.material_number ,
  b.department, 
  d.article_ft
  FROM POffer AS a
  JOIN ArticleHier AS b  --join article from PO with article hierarchy
     ON a.material_number = ltrim('0', b.article_number)
  JOIN Article AS d --join with article to obtain the name
    ON ltrim('0', d.article_number) = ltrim('0', b.article_number)
  WHERE a.customer_public_hash = '{hashes}' '''

  df = spark.sql(query)

  ingr=df.select(df.article_ft).toPandas()['article_ft']
  foodArticles=list(ingr)
  main_list.append(foodArticles)  # Append the list to the main list


# COMMAND ----------

# MAGIC %md
# MAGIC ## GenAI Usecase

# COMMAND ----------

# DBTITLE 1,LLM Approach
import openai
import langchain_openai
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

language = "Greek"
number_of_new_ingredients = 2

llm4 = AzureChatOpenAI(
    openai_api_base= "https://oai-adp1reg-myrcp-dv-we-01.openai.azure.com/",
    openai_api_version= '2024-02-01',
    deployment_name="myrcp_dev",
    temperature=0,
    openai_api_key= dbutils.secrets.get(scope='myrcpscope1', key='myrcp-oai-primary-key'),
    openai_api_type = "azure",
)

# embeddings = OpenAIEmbeddings(deployment_id="text-embedding-ada-002", chunk_size=1)

# COMMAND ----------

# DBTITLE 1,Referring LLM
all_recipes = []  # Create an empty main list

for sublist in main_list:   

    response = llm4(
    [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=f"Πρότεινε μου μία γνωστή συνταγή με τα περισσότερα από αυτά τα συστατικά:{sublist}. Μπορείς επίσης να χρησιμοποιήσεις συστατικά που υπάρχουν ήδη στις περισσότερες κουζίνες. Μπορείς επίσης να προσθέσεις συστατικά {number_of_new_ingredients} που δεν περιλαμβάνονται στη λίστα. Μπορείς να χρησιμοποιήσεις το συντομευμένο όνομα των συστατικών στη συνταγή")
    ])
    
    output = (response.content)
    all_recipes.append(output)  # Append the list to the main list

# COMMAND ----------

# DBTITLE 1,CustomerHash- Pandas to DataFrame
# Create a pandas DataFrame for each recipe
cust_df = pd.DataFrame(list_of_hashes, columns=['customer_hash'])

# Add a unique ID to each row
cust_df['c_id'] = range(1, len(cust_df) + 1)

c_df = spark.createDataFrame(cust_df)
c_df.createOrReplaceTempView("cust_hash")
display(c_df)

# COMMAND ----------

# DBTITLE 1,Recipe - Pandas to DataFrame
# Create a pandas DataFrame for each recipe
recipe_df = pd.DataFrame(all_recipes, columns=['recipe'])

# Add a unique ID to each row
recipe_df['r_id'] = range(1, len(recipe_df) + 1)

# Convert the pandas DataFrame to a PySpark DataFrame
df = spark.createDataFrame(recipe_df)

# Add the 'timestamp' column
df = df.withColumn("timestamp", lit(current_timestamp()))
df = df.select('r_id','timestamp','recipe')
df.createOrReplaceTempView("recipe")

# Show the DataFrame
display(df)


# COMMAND ----------

# DBTITLE 1,Combining the 2 DF's - Cust & Recipe
common_df = spark.sql("""select c.customer_hash,r.recipe, r.timestamp from recipe r join cust_hash c on r.r_id=c.c_id""")
display(common_df)

# COMMAND ----------

# DBTITLE 1,Delta Files Write Location in ADLS
delta_table_path = f"{root_path}{opco}/my_recipe/"
print(delta_table_path)

# COMMAND ----------

# DBTITLE 1,Write to Delta
common_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("path", delta_table_path) \
    .save()

# COMMAND ----------

# DBTITLE 1,Create Delta Table
df = spark.sql(f"""
CREATE TABLE IF NOT EXISTS Prepared_myrecipe_GR(
  `customer_hash` STRING,
  `recipe` STRING,
  `timestamp` TIMESTAMP
)
USING delta
LOCATION '{delta_table_path}'
""")

# COMMAND ----------

# DBTITLE 1,Select the Delta Table
# MAGIC %sql
# MAGIC select * from Prepared_myrecipe_GR;

# COMMAND ----------


