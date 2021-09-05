![header](https://github.com/ThuanPhong0126/Question-Answering-XLM-Roberta-GermanQuAD/blob/master/img/header.jpeg)
# Question answering task in German with a combination of the GermanQuAD dataset and the XLM-Roberta model

## About the question answering task

The question answering task is a classic problem in natural language processing and has received much attention with great success. 
The task requires computers to read and understand and answer human questions through a source of knowledge. 
The task makes the distance between people and computers closer and helps people in many areas of life.

We build a question answering model in the German language, with input as a question and a context, after computer processing, the output is the answer with the answer extracted from the context in the input. 
Our model is adapted from the large version pre-trained XLM-Roberta model [[2]](#2) with the base configuration provided by Huggingface [[3]](#3). 
The dataset used by us for fine-tuning the model is the GermanQuAD dataset [[1]](#1). Example for the task:

- ***Input***
  - **Context**: Insbesondere bei Tests, die häufig wiederholt werden, ist deren Automatisierung angeraten. Dies ist vor allem bei Regressionstests und bei testgetriebener Entwicklung der Fall. Darüber hinaus kommt Testautomatisierung bei manuell nicht oder nur schwer durchführbaren Tests zum Einsatz (z. B. Lasttests). Durch Regressionstests wird nach Softwareänderungen meist im Zuge des System- oder Abnahmetests der fehlerfreie Erhalt der bisherigen Funktionalität überprüft. Bei der testgetriebenen Entwicklung werden die Tests im Zuge der Softwareentwicklung im Idealfall vor jeder Änderung ergänzt und nach jeder Änderung ausgeführt.Bei nicht automatisierten Tests ist in beiden Fällen der Aufwand so groß, dass häufig auf die Tests verzichtet wird.</li>
  - **Question**: Wozu dienen Regressionstests beim Testen von Software?
- ***Output***
  - **Answer**: Durch Regressionstests wird nach Softwareänderungen meist im Zuge des System- oder Abnahmetests der fehlerfreie Erhalt der bisherigen Funktionalität überprüft.

## Fine-tuning the model

Install the necessary libraries
```python
!pip install transformers
!pip install datasets
```

Fine-tuning the model
```python
!python run_qa.py \
  --model_name_or_path xlm-roberta-large \
  --train_file <path-to-train-file> \
  --validation_file <path-to-test-file> \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 4  \
  --learning_rate 2e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --max_answer_length 300 \
  --doc_stride 128 \
  --save_steps 1000 \
  --overwrite_output_dir \
  --output_dir  './XLMRoberta-finetuned-GermanQuAD'
```

Evaluate the model: Use file evaluation_germanquad.py to evaluate performance of model.

## Result
Accuracy on the test set with a couple of evaluate metrics. 
|       Model          |F1-score     |Exact Match|
|------------------|-----------|-----------|
|XLM-ROBERTA large |85.46|71.42|
|Keep testing other methods soon|...|...|

# My contact information
Ho Chi Minh city, Vietnam
- *Facebook*: https://www.facebook.com/thuan.phong.1801/
- *Gmail*: thuanphong180100@gmail.com
- *Twitter*: https://twitter.com/ThuanPhong15

# References
<a id="1">[1]</a>
Timo Möller, Julian Risch, and Malte Pietsch. (2021). GermanQuAD and GermanDPR: Improving Non-English Question Answering and Passage Retrieval.

<a id="2">[2]</a>
Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. (2020). Unsupervised Cross-lingual Representation Learning at Scale.

<a id="3">[3]</a>
https://huggingface.co/
