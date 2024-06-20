import json
import pandas as pd
from pandas.io.json import json_normalize


# The data is a JSON file containing ratings and reviews for a coffee shop.
# 

with open("blue-bottle-coffee-san-francisco.json", "r") as f:
     data = json.load(f)


# Let's extract the contents of reviewList (i.e. ratings and content) into a Pandas dataframe.
# 

df = json_normalize(data, "reviewList")
df.head(3)


# Since ratings were originally strings, let's convert them to numeric values so that we can do analyses on them.
# 

df.dtypes


df["ratings"] = pd.to_numeric(df["ratings"])
df.dtypes


df["ratings"].describe()


# Now that our Pandas dataframe is in the correct format, let's write it to BigQuery. The `df.to_gcp()` function below creates a dataset named `mydataset` and a table named `mytable` whose schema is `df.dtypes`. You may check that this dataset is present in the [Bigquery UI](https://cloud.google.com/bigquery/quickstart-web-ui#create_a_dataset).
# 

project_id = "your-project-ID"
df.to_gbq("mydataset.mytable", project_id=project_id, verbose=True, if_exists="replace")


# You may also query this dataset from within Pandas, which returns a dataframe with the query results.
# 

query = "SELECT * FROM mydataset.mytable LIMIT 5"
pd.read_gbq(query=query, dialect="standard", project_id=project_id)


