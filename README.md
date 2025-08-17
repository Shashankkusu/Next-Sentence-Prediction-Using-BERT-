# 📘 Next Sentence Prediction using BERT

---

## 🚀 Project Description

This project demonstrates how to use the BERT model for the NLP task of **Next Sentence Prediction (NSP)**. Next Sentence Prediction is a core pre-training task for BERT, enabling models to better understand the relationship between sentence pairs. Here, we guide you through loading a pre-trained BERT model, preparing data, and making NSP predictions on sample text using Python and Hugging Face Transformers.

---

## ✅ Features

- Implementation of BERT for Next Sentence Prediction (NSP)
- Preprocessing and formatting text data for BERT input
- Tokenization and encoding using BERT's tokenizer
- Training and evaluation on sample text data
- Demonstration of NSP predictions with example outputs

---

## 🛠️ Tech Stack

- **Python**
- **Jupyter Notebook**
- **[Hugging Face Transformers](https://huggingface.co/transformers/)**
- **PyTorch**
- **NumPy**, **Pandas**
- **Matplotlib/Seaborn** (for data visualization, if used)

---

## 🏗 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Shashankkusu/Next-Sentence-Prediction-Using-BERT-.git
   cd Next-Sentence-Prediction-Using-BERT-
   ```

2. **(Optional) Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Jupyter Notebook**
   ```bash
   jupyter notebook
   ```
   Open the notebook `Next_Sentence_Prediction_using_BERT (1).ipynb` in your browser.

---

## 📝 Usage

Follow these steps to explore Next Sentence Prediction with BERT:

### 1. **Load the BERT Tokenizer and Model**
```python
from transformers import BertTokenizer, BertForNextSentencePrediction

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
```

### 2. **Encode Sentences for NSP**
```python
sentence_a = "The weather is nice today."
sentence_b = "I will go for a walk."

encoding = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt')
```

### 3. **Run NSP Prediction**
```python
outputs = model(**encoding)
logits = outputs.logits
import torch
prob = torch.softmax(logits, dim=1)
is_next_prob = prob[0][0].item()
not_next_prob = prob[0][1].item()
```

### 4. **Interpret Results**
```python
if is_next_prob > not_next_prob:
    print("Sentence B is likely to follow Sentence A (Is Next).")
else:
    print("Sentence B is NOT likely to follow Sentence A (Not Next).")
```

---

## 📊 Results & Examples

Example output from the notebook:

```
Sentence 1: The weather is nice today.
Sentence 2: I will go for a walk.
Prediction: Is Next (Probability: 0.92)

Sentence 1: The weather is nice today.
Sentence 2: The stock market crashed in 2008.
Prediction: Not Next (Probability: 0.81)
```

You can visualize probabilities or prediction results using matplotlib/seaborn for a more interactive experience.

---

## 📂 Project Structure

```
Next-Sentence-Prediction-Using-BERT-/
│
├── Next_Sentence_Prediction_using_BERT (1).ipynb   # Main Jupyter notebook
├── requirements.txt                                # Python dependencies
├── data/                                           # (Optional) Data files if any
└── README.md                                       # Project documentation
```

---

## 🤝 Contributing Guidelines

We welcome contributions! To get started:

1. **Fork** the repository
2. **Create a new branch** for your feature or fix
3. **Commit** your changes with clear messages
4. **Push** to your forked repository
5. **Open a Pull Request (PR)** describing your changes

Please ensure your code follows best practices and includes relevant tests or notebook cells where applicable.

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/) team for their open-source NLP tools
- Google Research for the BERT paper:  
  *Devlin, Jacob, et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding."* [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

---

Enjoy exploring Next Sentence Prediction with BERT! 🚀
