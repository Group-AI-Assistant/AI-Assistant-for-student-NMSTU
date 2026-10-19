import json
import urllib.request

url = "http://127.0.0.1:11434/api/embeddings"
data = json.dumps({"model": "nomic-embed-text", "prompt": "Привет"}).encode(
    "utf-8"
)

req = urllib.request.Request(
    url, data=data, headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode("utf-8"))
        print(" УСПЕХ! Вектор получен! Размер:", len(res["embedding"]))
except urllib.error.HTTPError as e:
    print(f"Ошибка HTTP: {e.code} - {e.read().decode('utf-8')}")
except Exception as e:
    print(f"Другая ошибка: {e}")