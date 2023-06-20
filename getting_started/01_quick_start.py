# Databricks notebook source
import sparknlp

from sparknlp.base import *
from sparknlp.annotator import *

from pyspark.ml import Pipeline

print("Spark NLP version", sparknlp.version())

spark

# COMMAND ----------

# DBTITLE 1,Using Pretrained Pipelines
from sparknlp.pretrained import PretrainedPipeline

pipeline_dl = PretrainedPipeline('explain_document_dl', lang='en')

# COMMAND ----------

# MAGIC %md
# MAGIC **Stages**
# MAGIC - DocumentAssembler
# MAGIC - SentenceDetector
# MAGIC - Tokenizer
# MAGIC - NER (NER with GloVe 100D embeddings, CoNLL2003 dataset)
# MAGIC - Lemmatizer
# MAGIC - Stemmer
# MAGIC - Part of Speech
# MAGIC - SpellChecker (Norvig)

# COMMAND ----------

testDoc = '''
Peter Parker is a very good person.
My life in RUssia is very interesting.
John and Peter are brothers. However they don't support each other that much.
Mercedes Benz is alos working on a driverless car.
Europe is very culture rich. There are huge churches! and big housed!
'''
result = pipeline_dl.annotate(testDoc)

# COMMAND ----------

result.keys()

# COMMAND ----------

import pandas as pd

df = pd.DataFrame({'token':result['token'], 'ner_label':result['ner'],
                      'spell_corrected':result['checked'], 'POS':result['pos'],
                      'lemmas':result['lemma'], 'stems':result['stem']})

df

# COMMAND ----------

# DBTITLE 1,Using fullAnnotate to get more details
detailed_result = pipeline_dl.fullAnnotate(testDoc)

detailed_result[0]['entities']

# COMMAND ----------

chunks=[]
entities=[]
for n in detailed_result[0]['entities']:
        
  chunks.append(n.result)
  entities.append(n.metadata['entity']) 
    
df = pd.DataFrame({'chunks':chunks, 'entities':entities})
df    

# COMMAND ----------

tuples = []

for x,y,z in zip(detailed_result[0]["token"], detailed_result[0]["pos"], detailed_result[0]["ner"]):

  tuples.append((int(x.metadata['sentence']), x.result, x.begin, x.end, y.result, z.result))

df = pd.DataFrame(tuples, columns=['sent_id','token','start','end','pos', 'ner'])

df


# COMMAND ----------

# DBTITLE 1,Sentiment Analysis
sentiment = PretrainedPipeline('analyze_sentiment', lang='en')

# COMMAND ----------

result = sentiment.annotate("The movie i watched today was not a good one")
result['sentiment']

# COMMAND ----------

sentiment_imdb_glove = PretrainedPipeline('analyze_sentimentdl_glove_imdb', lang='en')

# COMMAND ----------

comment = '''
It's a very scary film but what impressed me was how true the film sticks to the original's tricks; it isn't filled with loud in-your-face jump scares, in fact, a lot of what makes this film scary is the slick cinematography and intricate shadow play. The use of lighting and creation of atmosphere is what makes this film so tense, which is why it's perfectly suited for those who like Horror movies but without the obnoxious gore.
'''

result = sentiment_imdb_glove.annotate(comment)
result['sentiment']

# COMMAND ----------

# DBTITLE 1,Using the modules in a pipeline for custom tasks
!wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/jupyter/annotation/english/spark-nlp-basics/sample-sentences-en.txt

# COMMAND ----------

# dbutils.fs.cp("file:/databricks/driver/sample-sentences-en.txt", "dbfs:/")
display(dbutils.fs.ls("file:/databricks/driver/"))
