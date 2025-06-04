from fastapi import FastAPI
from pydantic import BaseModel
import fasttext
import re
# ссылка на модель: https://disk.yandex.ru/d/ZUtqGu8FiaMT2w
model = fasttext.load_model("best_fasttext_model.bin")

app = FastAPI(title="Tweet Sentiment Classifier")


class InputText(BaseModel):
    text: str

# Предобработка текста
def cleaner(documents):
    docs = []
    for doc in documents:
        text = doc.lower()
        text = re.sub(r"(@\w+)|(#\w+)", " ", text)
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"\w+://\S+", " ", text)
        text = re.sub(r"rt ", " ", text)
        text = re.sub(r" rt ", " ", text)
        text = re.sub(r":\(", " bad_flag ", text)
        text = re.sub(r"\(+", " bad_flag ", text)
        text = re.sub(r"99+", " bad_flag ", text)
        text = re.sub(r"0_0", " bad_flag ", text)
        text = re.sub(r"o_o", " bad_flag ", text)
        text = re.sub(r"о_о", " bad_flag ", text)
        text = re.sub(r":-\(", " bad_flag ", text)
        text = re.sub(r"=\(", " bad_flag ", text)
        text = re.sub(r" \(", " bad_flag ", text)
        text = re.sub(r";\)", " good_flag ", text)
        text = re.sub(r":d+", " good_flag ", text)
        text = re.sub(r"\=+\)+", " good_flag ", text)
        text = re.sub(r"\)+", " good_flag ", text)
        text = re.sub(r":\)", " good_flag ", text)
        text = re.sub(r"[^\w\s]", "", text)
        docs.append(text.strip())
    return docs

@app.post("/predict/")
def predict(input_data: InputText):
    cleaned_text = cleaner([input_data.text])[0]
    label, confidence = model.predict(cleaned_text)
    readable_label = "positive" if label[0] == "__label__1" else "negative"
    return {"label": readable_label, "confidence": round(float(confidence[0]), 4)}
