import requests
import time

BASE = "http://wmbprojects.com:42068/paraphrase"

t0 = time.time()
response = requests.put(BASE, {"sentence": "Programs must be written for people to read, and only incidentally for machines to execute."})
t1 = time.time()
print(t1-t0)
for s in response.json()["data"]:
    print(s)