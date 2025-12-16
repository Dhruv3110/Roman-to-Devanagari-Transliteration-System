# Roman-to-Devanagari Transliteration System 🇮🇳

A robust **Hinglish (Roman Hindi) → Hindi (Devanagari)** transliteration system trained using a **fine-tuned Gemma-2B model with LoRA**, designed to handle **slang, noisy spellings, abbreviations, and repeated characters** commonly found in informal text.

---

## Motivation

Hinglish text (e.g., *"mujhe jana h"*) is widely used in social media, chats, and informal communication, but most NLP systems expect **standard Hindi (Devanagari)**.

This project aims to:
- Convert noisy Roman Hindi into correct Devanagari Hindi
- Handle spelling variations, slang, and informal patterns
- Build a **robust, scalable, and evaluation-driven** transliteration system

---

## Features

- Hinglish → Hindi (Devanagari) transliteration
- Noise-aware preprocessing & scoring
- Data augmentation for robustness
- Fine-tuned **Gemma-2B** using **QLoRA (4-bit)**
- Character-level evaluation (CER, chrF)
- Error analysis vs noise score
- GitHub-friendly notebook with visible outputs

---

## Model & Training Details

| Component | Description |
|--------|------------|
| Base Model | `google/gemma-2b-it` |
| Fine-tuning | LoRA (QLoRA, 4-bit quantization) |
| Framework | Unsloth + HuggingFace |
| Sequence Length | 512 |
| Batch Size | 2 (with gradient accumulation) |
| Optimizer | AdamW |
| Training Type | Supervised Fine-Tuning (SFT) |

---

## Dataset

- **Dataset**: Hinglish–Hindi Transliteration Dataset  
- **Source**: HuggingFace  
- **Fields**:
  - `roman`: Hinglish (Roman Hindi)
  - `hindi`: Ground-truth Devanagari Hindi

### Data Processing
- Lowercasing & Unicode cleaning
- Removal of invalid characters
- Noise scoring based on:
  - Slang tokens
  - Repeated characters
  - Consonant-only tokens
  - Punctuation noise
- Rule-based data augmentation

---

## Prompt Format

```text
### Instruction:
Convert Hinglish (Roman Hindi) into Hindi (Devanagari script).
Be robust to slang, shortcuts, repeated letters & noisy spellings.

### Input:
mujhe jana h

### Output:
मुझे जाना है
```
## Abstract

The increasing use of Romanized typing for Indo-Aryan languages on social media presents significant challenges due to the lack of standardization, frequent phonetic shorthand, and loss of linguistic richness. While existing transliteration models perform well on structured datasets, they struggle with the noisy and informal nature of real-world digital communication, which is dominated by abbreviations, acronyms, vowel omissions, and inconsistent spellings.

To address this gap, this project fine-tunes the **Gemma-2B-IT model using 4-bit QLoRA** on a strategically augmented Hinglish–Hindi transliteration dataset. The augmentation process explicitly incorporates social-media-style noise patterns to improve robustness. The proposed sentence-level back-transliteration approach effectively resolves contextual ambiguities in Romanized Hindi and significantly improves performance on short, noisy inputs. As a result, the system offers a more practical and reliable solution for converting informal Romanized Hindi into native Devanagari script.

---

## Introduction

The widespread adoption of social media platforms and the dominance of English keyboards have led to a surge in the use of Romanized typing for Indo-Aryan languages. While convenient for informal communication, Romanized text lacks standardization and exhibits significant spelling variation, phonetic inconsistencies, and frequent vowel omissions.

For example, the Hindi word **नमस्ते** may appear as *Namste*, *Nmst*, or *Namastey*. Additionally, Romanized text often introduces contextual ambiguity, where the same Roman token can map to multiple Devanagari forms depending on usage. Such inconsistencies negatively impact both human understanding and NLP systems such as machine translation and sentiment analysis.

Romanized scripts also fail to preserve linguistic richness and phonetic distinctions present in native scripts. Critical contrasts—such as retroflex and dental sounds (ट vs त)—are commonly collapsed into a single Roman representation. These challenges highlight the necessity of robust back-transliteration systems that can accurately map Romanized Hindi into Devanagari while accounting for real-world typing behavior.
<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/88941963-4a96-4dd9-ba31-d96fe27b9248" />

---

## Problem Definition

Existing transliteration models are primarily trained on structured and standardized datasets (e.g., Dakshina corpus). While effective on clean inputs, these models fail to capture the noisy and inconsistent characteristics of real-world Hinglish text found on social media and chat platforms.

Informal communication often involves:
- Non-standard spellings
- Heavy vowel omissions (e.g., *ky* instead of *kya*)
- Abbreviations and shorthand (e.g., *tm ho*)
- Extremely short inputs

As a result, model performance degrades sharply on such inputs, reflected in increased **CER**, **WER**, and reduced **BLEU** scores. This lack of robustness limits the applicability of existing transliteration systems in practical NLP applications such as chatbots, machine translation, and sentiment analysis.

---

## Solution Strategy

To overcome these limitations, the proposed approach focuses on **embracing noise rather than eliminating it** during training. The system is designed to reflect the informal, inconsistent nature of real-world Hinglish communication.

### Key Steps:
- **Soft Cleaning**: Removes irrelevant symbols while preserving natural informal spellings.
- **Noise Detection**: Identifies slang, vowel omissions, consonant-heavy abbreviations, and repeated characters.
- **Noise Scoring**: Assigns each sentence a noise score to model varying levels of distortion.
- **Noise-Aware Data Augmentation**:
  - Vowel simplification (e.g., *baat → bat*)
  - Phonetic substitutions (e.g., *shaadi → saadi*, *khana → kana*)
  - Repetition handling
- **Instruction-Based Prompting**: Converts samples into structured prompts emphasizing robustness to noise.

The augmented dataset is then used to fine-tune **Gemma-2B-IT** using **LoRA-based parameter-efficient fine-tuning**, allowing effective learning with limited computational resources. Model performance is evaluated using character-level and word-level metrics, along with manual testing on highly abbreviated real-world examples.

---

## Dataset Description

The **Hinglish–Hindi Transliteration Dataset** consists of paired sentences where Hindi written in Roman script is aligned with its corresponding Devanagari representation. The dataset is specifically designed for transliteration tasks rather than translation, making it suitable for script normalization applications.

### Use Cases:
- Typing assistants
- Search systems handling mixed-script input
- NLP pipelines requiring normalization of user-generated content

### Limitations:
- Relatively small size
- Limited representation of real-world noisy Hinglish
- Controlled examples lacking extensive slang and abbreviations

To mitigate these limitations, extensive noise-aware augmentation was applied to better simulate social-media-style text.

Before Augmentation
<p align = "center">
  <img width="600" height="660" alt="image" src="https://github.com/user-attachments/assets/1bd70691-6557-4d2d-9fdf-414f464f59f2" />
</p>

After Augmentation

<p align = "center">
  <img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/aa4a298e-6862-4df1-8ad6-46d6f85802ff" />
</p>

---
## Methodology
<p align = "center">
  <img width="600" height="800" alt="image" src="https://github.com/user-attachments/assets/b158406b-aed0-443e-8b08-4753eec619d9" />
</p>

### Block Diagram Description
The block diagram illustrates the end-to-end workflow of the proposed noise-aware Hinglish-to-Hindi transliteration system. The process begins with the acquisition of a Hinglish–Hindi parallel dataset from the HuggingFace repository. This dataset serves as the foundational input for model development. The acquired data undergoes soft cleaning, where unnecessary symbols and non-linguistic characters are removed while retaining informal spellings that reflect real conversational behavior.

Following preprocessing, a noise detection mechanism is applied to categorize different forms of informal linguistic patterns commonly found in digital communication, including abbreviations, informal tokens, vowel-omitted words, and character repetition. Each sentence is evaluated based on these features, and a noise score is assigned to quantify the degree of noise present in the text.

The next stage implements noise-aware data augmentation, which generates realistic spelling variations conditioned on the calculated noise score. This augmentation process incorporates transformations such as vowel length variations and phonetic spellings, enabling the model to learn from multiple representations of noisy Hinglish inputs. The augmented dataset is then separated into training and validation subsets to facilitate experimental evaluation.

The training subset is used to fine-tune the Gemma 2B-IT model using QLoRA, a parameter-efficient adaptation technique that optimizes model performance while reducing computational overhead. The fine-tuned model is subsequently evaluated using objective metrics including Word Error Rate (WER), Character Error Rate (CER), and BLEU, along with manual testing on real-world short and highly abbreviated user-style inputs.

The complete workflow enables the development of a transliteration system that is robust to noisy, informal Hinglish text, improving its applicability in real-world applications such as conversational AI tools, sentiment analysis systems, and multilingual communication platforms.


---

## Outcome

The experimental results demonstrate that the proposed Roman-to-Devanagari transliteration system achieves high accuracy and strong robustness when evaluated on noisy Hinglish text. The model records a very low Character Error Rate (CER) of 0.0428, indicating precise character-level mapping from Romanized Hindi to Devanagari script. This confirms the model’s effectiveness in handling spelling variations, vowel omissions, and phonetic inconsistencies commonly found in informal user-generated content.

The Word Error Rate (WER) of 0.0637 further highlights the system’s ability to preserve word-level correctness, even in short and highly abbreviated inputs. Additionally, a chrF score of 92.57 and a BLEU score of 87.15 reflect strong overlap between predicted and reference outputs, validating both structural and contextual accuracy of the transliterations.

Despite the strict nature of exact matching, the model achieves an Exact Match Accuracy of 64.0%, which is notably high for noisy transliteration tasks where multiple valid spellings may exist. Furthermore, the Sentence-level Character Accuracy of 95.21% indicates that most predictions are either fully correct or contain only minor character-level deviations.

Overall, these results confirm that the noise-aware training strategy and data augmentation approach significantly improve the model’s real-world applicability. The system demonstrates reliable performance on informal Hinglish text and is well-suited for practical NLP applications such as chatbots, sentiment analysis, text normalization, and machine translation pipelines that rely on accurate processing of user-generated content.

<p>
  <img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/4935d22a-b7b0-40b9-9b7e-f3801bd1b3c1" />  
</p>


---

## Evaluation Metrics

Evaluation focuses on **character-level accuracy**, which is the most suitable metric for transliteration tasks, where minor spelling variations are acceptable.

| Metric | Score |
|------|------|
| **CER (↓)** | **0.0428** |
| **WER (↓)** | 0.0637 |
| **chrF (↑)** | **92.57** |
| BLEU | 87.15 |
| Exact Match | 64.0% |
| Sentence-level Char Accuracy | 95.21% |


