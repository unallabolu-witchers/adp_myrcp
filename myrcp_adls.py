# Databricks notebook source
# DBTITLE 1,List Scope
# display(dbutils.secrets.list(scope='myrcpscope1'))

# COMMAND ----------

# DBTITLE 1,ADLS Configuration
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

# DBTITLE 1,Root Path Define
root_path = f"abfss://myrcp@saadp1regmyrcpdvwe01.dfs.core.windows.net/"
root_files = dbutils.fs.ls(root_path)
# display(root_files)

# COMMAND ----------

PO_path = f"{root_path}RS/PersonalOffers/{DateToProcess_path}/offer_allocations_RS_{DateToProcess_filename}.csv"

Article_Path = f"{root_path}RS/Article/DELTA_Identity/"

ArticleHierarchy_Path = f"{root_path}RS/ArticleHierarchy/DELTA_Identity/"

# COMMAND ----------

# DBTITLE 1,Read the Files from ADLS
POfferDf = spark.read.csv(f"{PO_path}",header=True)
POfferDf.createOrReplaceTempView('POffer')

ArticleDf = spark.read.format("delta").load(Article_Path)
ArticleDf.createOrReplaceTempView('Article')

ArticleHierDf = spark.read.format("delta").load(ArticleHierarchy_Path)
ArticleHierDf.createOrReplaceTempView('ArticleHier')
