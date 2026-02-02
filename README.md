# Amazon ML Challenge 2025: 3 Days, No Sleep, and a Lot of ML 

**Final Score:** 45.0 SMAPE  
**Status:** (Survived)

## The Backstory
This repository is a collection of notebooks; along with the result of a 72-hour sprint. I teamed up with Master's students from IIT Madras, and we went all in. 

We barely slept (4 AM to 8:30 AM was our "night"), and constantly coordinated over calls. We didn't win the grand prize, but the experience of building a massive multimodal pipeline from scratch in three days was honestly a win in itself.

## The Team & The Hardware
This was a split effort. While I focused on the Vision (Images) and tabular data, my teammates handled the heavy lifting on the Text side.

* **Me:** Handled the Image pipeline (Vision Transformers), Data Cleaning, and the final Ensemble models.
* **The IIT Madras Squad:** Handled the intense Text Feature Extraction (BERT/LLM embeddings). They actually rented **NVIDIA RTX 4090 on Vast.ai** for the text data because the dataset was massive.

## Our Approach

We treated this as a true **Multimodal Regression** problem. We didn't just want to predict price; we wanted the model to *understand* the product.

### 1. The Strategy
We realized that `SMAPE` (the competition metric) cares about relative error. So, instead of predicting the raw price, we predicted `log(1 + price)`. This stabilized the training and aligned the Loss function with the evaluation metric.

There are:

- 75,000 training samples

- 75,000 test samples

We did a full data + image audit to make sure nothing was broken:

Only 1 image missing in train and test

No corrupted images

Price was highly skewed, so we used log-transform

### 2. The Pipeline
* **Images (My Part):** I used a pre-trained **Vision Transformer (ViT-Base)** to extract 768-dimensional embeddings for every single product image. This helped distinguish between a "single bottle" vs. a "pack of 6" just by looking at the image context.
* **Text (Team Part):** My teammates ran the product titles and descriptions through heavy Transformer models on the cloud GPUs to generate dense vector embeddings.
* **Tabular:** We used Regex to extract hidden details like "Pack of X", volume (oz, ml), and brand names.

### 3. The Model
We fed all these features (2000+ columns!) into a weighted ensemble of **LightGBM** and **CatBoost**.

## 📂 Repository Structure

```text
├── notebooks/
│   ├── 01_Image_Feature_Extraction_ViT.ipynb   <-- My heavy lifting: ViT implementation
│   ├── 02_Text_Feature_Engineering.ipynb       <-- The Regex & Embedding work
│   ├── 03_Baseline_Pipeline.ipynb              <-- Our first attempt (EDA + Prototype)
│   ├── 04_Final_Ensemble_Modeling.ipynb        <-- The Master File (LGBM + CatBoost)
│   └── BLUNDER.ipynb                           <-- (See below... 😅)
```

## The "Big Blunder" (A Learning Moment)

I decided to keep a file named `BLUNDER.ipynb` in the repo because, well… it happened.

Midway through the competition, our validation score suddenly looked *too* good. For a moment I honestly thought we had cracked the code. But later I realized that when I was writing the custom SMAPE metric function, I had accidentally **divided by 2 instead of multiplying by 2**. That mistake effectively halved the error and gave us a fake "2x" score boost.

It was a pretty heart-dropping moment when I caught it. I fixed the metric but it was too late by then, It is 11:58pm 13th october 2025 just 2 min before the competition ends, but not able to make it, so I accepted the real score, and moved on. I kept the file in the repository as a reminder to myself and anyone reading this: always double-check your evaluation metric, because a small math mistake can completely mislead you.
