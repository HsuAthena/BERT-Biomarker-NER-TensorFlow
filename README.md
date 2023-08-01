## Biomarker-BERT-NER
Named Entity Recognition (NER) model to extract biomarker names from notes.

Goal: The objective is to extract biomarker names from medical notes. Biomarkers play a crucial role in cancer care by enabling early detection, personalizing cancer treatments, and monitoring treatment responses. To accurately extract these biomarkers from the notes, I built a TensorFlow BERT NER model specifically to identify the biomarker names.

Technique used in this model: Given that I only have around 500 annotations, which is a relatively small dataset, I incorporated K-fold cross-validation to assess the model's performance. Additionally, I implemented EarlyStopping to prevent overfitting. 

Model: emilyalsentzer/Bio_ClinicalBERT from Hugging Face