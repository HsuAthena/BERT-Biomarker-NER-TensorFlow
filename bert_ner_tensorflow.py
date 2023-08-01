from transformers import AutoTokenizer, AutoModel
from itertools import groupby
import json
import tensorflow as tf
from transformers import TFAutoModelForTokenClassification, create_optimizer
from transformers import DataCollatorForTokenClassification
from datasets import load_metric
import numpy as np
from transformers import pipeline
from datasets import load_dataset, load_metric, ClassLabel
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from sklearn.model_selection import KFold


model_id = "emilyalsentzer/Bio_ClinicalBERT"
seed=33
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
id2label = {0:'O',1:'B-BIOMARKER',2:'I-BIOMARKER'} 
label2id = {'O':0,'B-BIOMARKER':1,'I-BIOMARKER':2}
Earlystopping= [EarlyStopping(monitor='val_loss', patience=5, mode='min', min_delta=0.0001)]
checkpoint_path='/content/drive/MyDrive/Colab Notebooks/BiomarkerTfBERT_NER'
dataset_filename = '/home/yhou/temp/athena/revised/Allannotate_biomarker.json'

num_train_epochs = 20
train_batch_size = 3
eval_batch_size = 3
learning_rate = 2e-5 
weight_decay_rate=0.01
num_warmup_steps=0
#k fold
split_number=5 
# eval p,r,f1
metric = load_metric("seqeval")

dataset_All = load_dataset('json', data_files=dataset_filename)

def tokenize_and_align_labels_MIC(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]): 
        # get a list of tokens their connecting word id (for words tokenized into multiple chunks)
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to the current
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets_All= dataset_All.map(tokenize_and_align_labels_MIC, batched=True)
pre_tokenizer_columns_All = set(dataset_All["train"].features)
tokenizer_columns_All = list(set(tokenized_datasets_All["train"].features) - pre_tokenizer_columns_All)


#K fold 
kf = KFold(n_splits=split_number)
bmkr_P=[]
bmkr_F1=[]
bmkr_R=[]
Ttl_P=[]
Ttl_R=[]
Ttl_F1=[]
Ttl_A=[]
for train_index, test_index in kf.split(tokenized_datasets_All["train"]):
  Train_datasetlist=[]
  Test_datasetlist=[]
  train_index=train_index.tolist()
  test_index=test_index.tolist()
  for x in train_index:
    Train_datasetlist.append(tokenized_datasets_All["train"][x])
  for i in test_index:
    Test_datasetlist.append(tokenized_datasets_All["train"][i])
  
  # tempoaray train file, create dataset format, when k fold looping, it will replce the origianl
  TempTrain_filename = '/home/yhou/temp/athena/revised/temp_train.json'
  # tempoaray test file. 
  TempTest_filename = '/home/yhou/temp/athena/revised/temp_test.json'
  with open(TempTrain_filename, 'w') as outfile:
    for dataset in Train_datasetlist:
      outfile.write(json.dumps(dataset) + '\n')


  with open(TempTest_filename, 'w') as outfile:
    for dataset in Test_datasetlist:
      outfile.write(json.dumps(dataset) + '\n')



  KTrain_dataset = load_dataset('json', data_files=TempTrain_filename)
  KTest_dataset = load_dataset('json', data_files=TempTest_filename)

  
  # Data collator that will dynamically pad the inputs received, as well as the labels.
  data_collator_All = DataCollatorForTokenClassification(
    tokenizer=tokenizer, return_tensors="tf"
  )

# converting our train dataset to tf.data.Dataset
  Ktf_train_dataset = KTrain_dataset['train'].to_tf_dataset(
    columns= tokenizer_columns_All,
    shuffle=False,
    batch_size=train_batch_size,
    collate_fn=data_collator_All,
  )

# converting our test dataset to tf.data.Dataset
  Ktf_eval_dataset =KTest_dataset['train'].to_tf_dataset(
    columns=tokenizer_columns_All,
    shuffle=False,
    batch_size=eval_batch_size,
    collate_fn=data_collator_All,
  )

  # model train
  num_train_steps = len(Ktf_train_dataset) * num_train_epochs
  optimizer, lr_schedule = create_optimizer(
    init_lr=learning_rate,
    num_train_steps=num_train_steps,
    weight_decay_rate=weight_decay_rate,
    num_warmup_steps=num_warmup_steps,
  )

  model_K = TFAutoModelForTokenClassification.from_pretrained(
    model_id,
    id2label=id2label,
    label2id=label2id,
    from_pt=True
  )

  model_K.compile(optimizer=optimizer)
  model_K.fit(
    Ktf_train_dataset,
    validation_data=Ktf_eval_dataset,
    epochs=num_train_epochs,callbacks=Earlystopping
  )


  # Evaluation
  ner_labels=list(model_K.config.id2label.values())
  all_predictions = []
  all_labels = []
  for batch in Ktf_eval_dataset:
      logits = model_K.predict(batch)["logits"]
      labels = batch["labels"]
      predictions = np.argmax(logits, axis=-1)
      for prediction, label in zip(predictions, labels):
          for predicted_idx, label_idx in zip(prediction, label):
              if label_idx == -100:
                  continue
              all_predictions.append(ner_labels[predicted_idx])
              all_labels.append(ner_labels[label_idx])
  print(metric.compute(predictions=[all_predictions], references=[all_labels]))
  result=metric.compute(predictions=[all_predictions], references=[all_labels])



  bmkr_P.append(result['BIOMARKER']['precision'])
  bmkr_R.append(result['BIOMARKER']['recall'])
  bmkr_F1.append(result['BIOMARKER']['f1'])
  Ttl_P.append(result['overall_precision'])
  Ttl_R.append(result['overall_recall'])
  Ttl_F1.append(result['overall_f1'])
  Ttl_A.append(result['overall_accuracy'])

biomarker_Precision=sum(bmkr_P)/split_number
biomarker_Recall=sum(bmkr_R)/split_number
biomarker_F1=sum(bmkr_F1)/split_number
Total_Precision=sum(Ttl_P)/split_number
Total_Recall=sum(Ttl_R)/split_number
Total_F1=sum(Ttl_F1)/split_number
Total_Accuracy=sum(Ttl_A)/split_number
print("Avg Biomarker Precision: ",biomarker_Precision," Recall: ",biomarker_Recall," F1: ",biomarker_F1)
print("Avg Total Precision: ",Total_Precision," Recall: ",Total_Recall," F1: ",Total_F1, " Accuracy ",Total_Accuracy)

# save the model
model_K.save_pretrained(checkpoint_path)
