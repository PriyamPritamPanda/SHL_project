# 🎧 SHL_project

This repository contains two different approaches to solve the **grammar scoring task** for spoken audios. The goal is to assign a **grammar score between 0 and 5** to each `.wav` audio file based on the grammatical quality of the speech.

---

## 📌 Task Description

Given a dataset of `.wav` audio files and their associated grammar scores, the task is to build a model that can predict the grammar score for unseen audio files. The final predicted score must lie between **0 and 5**, as per the competition rule.

---

## 🚀 Approaches

### 🔉 Approach 1: Wav2Vec2 + XGBoost Regressor

- **Objective**: Predict the grammar score directly from raw audio.
- **Steps**:
  1. Used the `Wav2Vec2` model to extract 768-dimensional embedding vectors from each audio file.
  2. Trained an `XGBoost Regressor` model on the embeddings.
  3. Final predictions were clipped between 0 and 5.

- **File**: `wav2vec2-xgb-grammar-scoring.ipynb`

---

### 🗣️ Approach 2: Whisper + BERT + MLP

- **Objective**: Predict the grammar score from the transcribed text.
- **Steps**:
  1. Transcribed each audio file using OpenAI's `Whisper` model.
  2. Passed the transcriptions through `bert-base-uncased` to generate 768-dimensional sentence embeddings.
  3. Trained an **MLP** consisting of:
     - One hidden layer with 256 neurons and ReLU activation.
     - An output layer with a sigmoid activation.
     - The final output was scaled by multiplying by 5 to ensure scores between 0 and 5.

- **File**: `bert-whisper2 (1).ipynb`

---

## 📁 File Structure

