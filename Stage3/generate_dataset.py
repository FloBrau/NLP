import random
import csv
import re
import os
from pathlib import Path
from PyPDF2 import PdfReader
from transformers import pipeline
from transformers import PegasusForConditionalGeneration, PegasusTokenizerFast

report = "articles/IPCC_AR6_SYR_FullVolume.pdf"
output = "train.csv"

#https://huggingface.co/tuner007/pegasus_paraphrase
def paraphrase(sentence):
    inputs = tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt")
    outputs = model.generate(
        **inputs,
        num_beams=5,
        num_return_sequences=1
    )
    #print(sentence)
    #print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

def contradict(sentence):
    if " is " in sentence:
        return sentence.replace(" is ", " is not ")
    if " are " in sentence:
        return sentence.replace(" are ", " are not ")
    if " was " in sentence:
        return sentence.replace(" was ", " was not ")
    if " were " in sentence:
        return sentence.replace(" were ", " were not ")

    numbers = re.findall(r'\d+', sentence)
    if numbers:
        number = numbers[0]
        return sentence.replace(number, str(int(number) + random.randint(10, 50)))

    return "It is false that " + sentence


#model = "Vamsi/T5_Paraphrase_Paws" # didnt work; just producded same sentence as input
model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
tokenizer = PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase")

sentences = []

reader = PdfReader(report)
text = ""

for page in reader.pages:
    text += page.extract_text() + " "

sentences = re.split(r'(?<=[.!?]) +', text)
sentences = [s.strip() for s in sentences if 50 < len(s) < 250]

data = []
for i in range(500):
    premise = random.choice(sentences)

    paraphrased = paraphrase(premise)
    data.append([premise, paraphrased, "entailment"])

    #contradiction = contradict(premise)
    contradiction = contradict(paraphrased)
    data.append([premise, contradiction, "contradiction"])

    neutral = random.choice(sentences)
    data.append([premise, neutral, "neutral"])

    if i % 10 == 0:
        print(f"Sentence: {premise}")
        print(f"Entail: {paraphrased}")
        print(f"Contra: {contradiction}")
        print(f"Neutral: {neutral}")

with open(output, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["premise", "hypothesis", "label"])
    writer.writerows(data)

print("Dataset generated!!")
