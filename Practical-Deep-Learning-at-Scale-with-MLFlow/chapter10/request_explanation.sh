curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{
    "dataframe_split": {
      "columns": ["text"],
      "data": [["This is the best movie we saw."], ["What a movie!"]]
    }
}'
