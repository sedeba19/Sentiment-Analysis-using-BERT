# Sentiment-Analysis-using-BERT

Distil BERT, a distilled version of BERT: smaller, faster, cheaper and lighter

This is from paper: https://arxiv.org/abs/1910.01108

As transfer learning from large-scale pre-trained models becomes more prevalent in Natrual Language Processing (NLP), operating these large models in on the edge and/or under constrained computational training or inference budgets remains challenging. In this work, we prospose a method to pre-train a smaller general-purpose language representation model, called DistilBERT, which can then be fine-tuned with good performances on a wide range of tasks like its larger counterparts. While most prior work investigated the use of distillation for building task-specific models, we leverage knowledge distillation during the pre-training phase and show that it is possible to reduce the size of a BERT model by 40%, while retaining 97% of its language understanding capabilities and being 60% faster. To leverage the inductive biases learned by larger models during pre-training, we introduce a triple loss combining language modeling, distillation and cosine-distance losses. Our smaller, faster and lighter model is cheaper to pre-train and we demonstrate its capabilities for on-device computations in a proof of concept experiment and a comparative on-device study.

What is DistilBERT?

BERT is designed to pretrain deep bidrectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be finetuned with just one additional output layer to create state of the art models for a wide range of tasks, such as question answering and language inference, without substantial taskspecific architecture modifications.

DistilBERT is a small, fast, cheap and light Transformer model trained by distilling BERT base. It has 40% less parameters than bert-base-uncased, run 60% faster while preserving over 95% of Bert's performances as measured on the GLUE language understanding benchmark.

What is KTRAIN?

ktrain is a library to help build, train, debug and deploy neural networks in the deep learning software framework, Keras.

ktrain uses tf.keras in TensorFlow instead of standalone Keras. Inspired by the fastai library, with only a few lines of code, ktrain allows you to easily:
* estimate an optimal learning rate for your model given your data using a learning rate finder.
* employ learning rate schedules such as triangular learning rate policy, 1cycle policy, and SGDR to more effectively train your model.
* employ fast and easy to use pre-canned models for both text classification (e.g. NBSVM, fastText, GRU with pretrained word embeddings) and image classification (e.g. ResNet, Wide Residual Networks, Inception)
* load and preprocess text and image data from a variety of formats
* inspect data points that were misclassified to help improve your model
* leverage a simple prediction API for saving and deploying both models and data pre-processing steps to make predictions on new raw data

ktrain GitHub: https://github.com/amaiya/ktrain

