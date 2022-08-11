# FinLitQA

**Abstract:**

Machine Reading Comprehension is a crucial component of Natural Language processing. Many recent developments have led to the MRC systems attaining human-level results on pre-defined evaluation tasks. Questions Answering systems (QAS) is one of the important applications of MRC.  The availability of Natural language models pre-trained on large datasets and the applicability of transfer learning in NLP have led to the use of QAS in many real-world applications. A robust Question Answering system for the financial domain to address the issue of low financial literacy in the Indian context is proposed in this research. 
In recent years, many steps have been taken by the Government of India to improve financial literacy and financial inclusion in the country. This research provides a way to enhance these efforts by employing a Question answering system that can answer financial questions in natural language by extracting answers from the given documents. The study also introduces an annotated dataset FinlitQA, a labeled Question answering dataset in the SQuAD V2 format. BERT and RoBERTa models were taken as the baseline models and different performance improvement measures were applied to achieve better performance. With the combined benefits of Domain Adaptive Pre-training, Task Adaptive Fine-Tuning, and Few Samples Fine-Tuning the system was able to achieve more than 10 points increase in the Exact Match measures and 6 points increase in the F1 scores for the evaluation dataset when compared to the baseline model. 

**Aim and Objectives:**

The main aim of this research is to build a robust and accurate question answering system based on pre-trained, large NLP models to interpret queries in plain language, make inferences from a corpus of curated, in-domain, and context-specific material, and answer financial domain-based questions. The Question answering system is particularly aimed at the under-educated and the younger population of India who is newly inducted into the financial system.
To work towards the aim, the research objectives are formulated as follows: 
•	To study and investigate the inner workings of a large pre-trained NLP model like BERT and infer how to apply such a model for a specific downstream task, in this case, question answering. 
•	To experiment with fine-tuning the already trained model with a task-specific dataset.
•	To examine the effect of further fine-tuning of the model with in-domain dataset and few samples user dataset and compare the performances for different experiments. 
•	To build an end-to-end Question Answering System based on information retrieval and answer extraction to enable the system to be applied in the real world.

**Experiments:**


**Continual Pre-Trained language models (CPLM)**
For the first experiment, the Continual Pre-Trained language model of FinBERT was chosen as the model to be used. FinBERT (Araci, 2019) was a BERT-based model which was then further pre-trained on Reuters TRC2-Financial corpus (24M words). FinBERT is an NLP model pre-trained on financial data corpus to do sentiment analysis on financial text. The pre-trained model was used from the Hugging Face transformer library for the experiment.

**Domain Adaptive Pre-Training (DAPT)**
Domain adaptive Pre-Training is the continuation of language modeling of a pre-trained large model with an in-domain data corpus. This Domain adaptive pre-training is done to enable the model to learn the language nuances from in-domain data. This can be highly beneficial in highly specialized requirements like medical domain, banking domain, etc. This process is also sometimes referred to as pre-finetuning. 

For the Domain adaptive Pre-Training, the BertForMakedLanguageModeling model was used. The DAPT was done in two steps.  
1.	DAPT with data prepared from FiQA challenge dataset. 
2.	DAPT with data prepared from the downloadable content on the NCFE.org website. (NCFE, 2021a)

**Task Adaptive Fine-Tuning (TAFT)**
Task Adaptive Fine-Tuning is the task of preparing the model for the downstream NLP by adding the final layer to the model. The SQuAD V2 dataset is one of the benchmark datasets for Question answering and was used for this task. 
The models resulting from the preceding LM pre-training steps were all fine-tuned with the SQuAD dataset to create a model ready for the downstream Question answering task. For the basic models bert-base-uncased and roberta-base, the already available SQuAD V2 fine-tuned version was used in the final evaluation instead of fine-tuning again. 

**Few Samples Fine-Tuning (FSFT)**
The FinLitQA dataset created using the Haystack annotation tool was finally used for the Few samples fine-tuning. In Few Samples Fine-Tuning, domain data is used to create task-specific labeled data and is then used in fine-tuning.  
Documents from reliable and current banking websites were curated and chosen for creating the question answers pair for FSFT. 
The run_qa.py script available in the Hugging Face library was used for the fine-tuning with the custom dataset with the following parameters. The models were fine-tuned for 2 Epochs after which the validation metrics did not seem to improve.

```
  --evaluation_strategy steps \
  --save_strategy steps \
  --weight_decay 0.01 \
  --warmup_ratio 0.2 \
  --per_device_eval_batch_size 8 \
  --per_device_train_batch_size 8 \
  --overwrite_output_dir \
  --eval_steps 200 \
  --save_steps 200 \
  --learning_rate 2e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
```


**References:**
1. Araci, D., (2019) FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. [online] Available at: https://arxiv.org/abs/1908.10063v1 [Accessed 13 Oct. 2021].
2. Deepset-ai, (2020a) Haystack Annotation Tool. [online] haystack.deepset.ai. Available at: https://annotate.deepset.ai.
3. Deepset-ai, (2020b) Haystack Overview. [online] haystack.deepset.ai. Available at: https://haystack.deepset.ai/overview/intro.
4. FiQA, (2018) FiQA - 2018. Google.com. Available at: https://sites.google.com/view/fiqa/ [Accessed 9 Aug. 2021].
5. Hugging Face, (2021) Models - Hugging Face. [online] huggingface.co. Available at: https://huggingface.co/models [Accessed 23 Dec. 2021].
6. McCormick, C., (2020) Question Answering with a Fine-Tuned BERT. [online] Available at: https://mccormickml.com/2020/03/10/question-answering-with-a-fine-tuned-BERT/ [Accessed 26 Dec. 2021].
7. NCFE, (2021b) https://www.ncfe.org.in/. [online] ncfe.org.in. Available at: https://www.ncfe.org.in/ [Accessed 9 Aug. 2021].
8. Rajpurkar, P., (2017) The Stanford Question Answering Dataset. [online] GitHub. Available at: https://rajpurkar.github.io/SQuAD-explorer/ [Accessed 9 Aug. 2021].
9. https://github.com/yuanbit/FinBERT-QA
10. https://github.com/jamescalam/transformers


