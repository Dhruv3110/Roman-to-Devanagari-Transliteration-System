# ROMAN - TO - DEVANAGARI TRANSLITERATION SYSTEM 🇮🇳

A robust **Hinglish (Roman Hindi) → Hindi (Devanagari)** transliteration system built using a **fine-tuned Gemma-2B-IT model with QLoRA**, designed to handle slang, noisy spellings, abbreviations, and repeated characters commonly found in real-world informal text.

This system is trained on noisy Hinglish data and evaluated on both in-domain and out-of-domain Romanized Hindi to ensure generalization.

---

## MOTIVATION

Hinglish (e.g., “mujhe jana h”) is the dominant writing style on social media, chat apps, and online forums. However, most NLP pipelines, search engines, and downstream language tools expect **standard Hindi written in Devanagari**.

This creates a major mismatch between how users write and how systems process text.

This project aims to:
- Convert noisy Roman Hindi into accurate Devanagari Hindi
- Handle spelling variations, slang, abbreviations, and informal writing
- Build a **noise-robust, evaluation-driven transliteration pipeline**

---

## KEY FEATURES

- Hinglish → Hindi (Devanagari) transliteration
- Noise-aware preprocessing and normalization
- Rule-based data augmentation for social-media-style text
- Fine-tuned Gemma-2B-IT using QLoRA (4-bit quantization)
- Evaluation using CER, WER, chrF, and BLEU
- Testing on both training-style and external Romanized Hindi datasets
- Reproducible, GitHub-ready Jupyter Notebook

---

## MODEL & TRAINING DATA

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



### NOISE MODELLING & PREPROCESSING

The evaluation pipeline explicitly models Hinglish noise:
- Lowercasing and Unicode normalization
- Removal of invalid and special characters
- Noise scoring based on:
  - Slang tokens (e.g., yr, bht, plz, kr)
  - Repeated characters (goooood, plsss)
  - Consonant-only tokens (hn, kr, mt)
  - Punctuation and symbol noise

This allows detailed error analysis as a function of noise level, showing how performance degrades with increasing informal writing.

---

## PROMPT FORMAT

```text
### Instruction:
Convert Hinglish (Roman Hindi) into Hindi (Devanagari script).
Be robust to slang, shortcuts, repeated letters & noisy spellings.

### Input:
mujhe jana h

### Output:
मुझे जाना है
```

## ABSTRACT

The widespread typology in Romanized typing systems of Indo-Aryan languages has emerged as a major challenge for NLP regarding these system's non-standard orthography, phonetic short-hand and ubiquitous vowel removal. Although state-of-the-art transliteration models work well on structured corpora, they often perform poorly in the noisy and informal context of today’s digital communication. In this project, we present a strong sentence level back-transliteration model obtained by fine-tuning the Gemma model with LoRA (Parameter-Efficient Fine-Tuning). In order to account for the diversity of social media text, we used a specialized data augmentation pipeline by simulating phonetic variations, informal language substitutions and dropping vowels used in “noisy” Roman Hindi. The implemented approach successfully resolves context ambiguities of the Romanized Hindi. Results show a significant increase in accuracy of transliteration, especially for short, informal sequences. The model meets a realistic demand of translating digital vernacular into formal Devanagari script, linking their informal spoken style and standard linguistic description.

---

## INTRODUCTION

Due to the popularity of social media and the widespread use of English keyboards, there is an increase in Romanized typing for Indo-Aryan languages for informal communication, especially via texting. The Romanized text in social media additionally exhibit inconsistency, like the variation of spelling, phonetics and vowel removal. This inconsistency causes ambiguity in representing same word e.g. नमस्ते (Namaste) can appear as Namste, Nmst, or Namastey (ref 1). Moreover, Romanized text involves one-many mappings with the context-dependent as in Romanised text, sir could also be सर (head) while at some other point it refers to सर (Sir). These inconsistencies can result in communication errors among humans, as well being reflected in mistakes made by NLP applications such as machine translation.

Furthermore, apart from the problems of standardisation, Romanised scripts are also unable to retain rich linguistic and phonetic variations in mother scripts, thus butchering cultural and language expressions. Some sounds in Hindi and other Indo-Aryan languages are not unambiguously represented by Roman script; hence ambiguities exist. For instance, the Devanagari characters ट (retroflex T) and त (dental T) can both be Romanised as T or Ta, ignoring a significant difference between retroflex and dental sounds in the regular language. Also, English sounds are not directly translatable to Hindi phonetics. For example, the English sounds v and w are both typically transliterated as व (v) which may be confused especially with ब-like sounds. These constraints illustrate the limitations of interpreting Romanized text as means of meaningful conversation or to faithfully represent speech.
These difficulties highlight the need for strong back-transliteration systems to convert the Wikipedia Romanized Indo-Aryan text into native script. Back-transliteration converts Romanization text to its native script in terms of phonological modelling, considering non-standardization and input irregularity nature. Proper back-transliteration contributes to digital communication by preserving linguistic diversity, making the text easier to read, and preventing confusion. Moreover, it allows Romanized data to be used with existing automated/home grown systems like machine translation, text-to-speech, text mining and other areas where such functionality will make them much more effective.

### Approaches to Back-Transliteration: 

The **rule-based methods** use the phonetic mapping and linguistic rules which have to be manually formulated for converting Romanized script into the target one. These systems use character-to-character, or symbol-level phonetic correspondences according to pronunciation rules. While rule-based systems are interpretable and work well on clean, standard language inputs, they tend to fail with spelling deviations or creative abbreviations and vowel deletions often occurring in informal social media data with contexts that may not be precise. It is also difficult to maintain and scale up such systems because of the high level of linguistic expertise needed.
For instance, such a simple rule-based system can have ka → क, kha → ख and na → न.

<p align= "center">
  <img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/88941963-4a96-4dd9-ba31-d96fe27b9248" />
</p>

<p align = "center">
  Fig 1: Transliteration Mapping Table for Devanagari Characters
</p>

<img width="1001" height="274" alt="image" src="https://github.com/user-attachments/assets/92e44963-3ffe-4070-8523-db675945ff3f" />

<p align = "center">
  Fig 2: Devanagari Vowel and Special Character Transliteration in HK and ITRANS Systems
</p>

**Statistical Transliteration approach:** These models transliterate as a probabilistic character-sequence transformation estimated from aligned Roman–Devanagari pairs. These approaches may be more generalized than rule-based systems since they learn common spelling variations.
Example:

-	Input: namste
-	Output: नमस्ते (correct, due to learned alignment)

However, performance degrades when encountering rare or heavily abbreviated forms:

-	Input: ky tm aa rhe
-	Output:  क्य ट्म आ रह
-	Expected: क्या तुम आ रहे
  
Statistical methods can perform poorly on very short or consonant-rich input due to their underrepresentation in clean training dataset.

In the domain of **Machine Learning or Deep Learning Methods**, different Neural sequence-to-sequence models as Transformer-based architectures learn contextual character dependencies and show significant improvement in transliteration quality over structured datasets.

Example:

-	Input: tum ho
-	Output: तुम हो (correct)
  
Yet, these models still struggle with noisy informal text:

-	Input: tm ho
-	Output:  तम हो or टम हो
-	Expected: तुम हो

Similarly, vowel-dropping causes confusion:

-	Input: kr diya
-	Output:  कर दिया / क्र दिया (inconsistent)
-	Expected: कर दिया

This limitation is due to the fact that ML/DL models are in most of cases trained on pure or semi-pure dataset, and do not properly capture informal typing habits.

**LLM-based transliteration models** make use of pretrained language representations which somehow encode phonetics, semantics or both. Our instruction-tuned models can be more effective in predicting the noisy text.

Example:

-	Input: ky tm kl aaoge
-	Output: क्या तुम कल आओगे (correct)

LLMs also handle contextual disambiguation:

-	Input: sir dard ho rha h
-	Output: सिर दर्द हो रहा है (correct)

In this paper, we have used LLM-based Approach for back-transliteration


---

## PROBLEM DEFINITION

The present model for transliteration has mainly been tested with synthetic and structured datasets. Although these datasets are useful for having clean and regular examples for Roman to Devanagari conversion, they do not consider the low fidelity of actual digital data. In social media, chatting applications and informal contexts users often use non-standard written forms, creative acronyms and various shortcuts to represent words while avoiding typing all the characters in the word, particularly omitting vowels or replacing letters with numbers). As the training samples do not sufficiently represent such noisy variations, that model is woefully unready to deal with exactly this kind of input.

In addition to this limitation, the model demonstrates significant weaknesses when dealing with short texts and heavy vowel omissions. Informal communication is often characterized by brevity, where users prefer minimal input to maximize typing speed. For instance, instead of typing “kya,” users may write “ky,” or instead of “tum ho,” they may write “tm ho.” These kinds of inputs result in a sharp decline in accuracy for the system, as reflected in evaluation results where metrics such as Character Error Rate (CER), and CHRF scores show a considerable drop on datasets containing such noisy examples. This is an indication that the model is not very good at capturing casual way of communication which has become a standard behaviour in digital conversations these days.
All these issues combined hinder the use of the deck on realistic scenarios where it could be useful and real-world modularity support would be very helpful, especially in sentiment analysis, chatbots (which require precise user-generated content processing) or machine translation similar tasks.


---

## SOLUTION STRATEGY

To mitigate the limitations of existing transliteration models that are predominantly trained on clean and highly structured datasets, our approach adopts a training strategy that closely mirrors the informal and noisy nature of real-world Romanized Hindi communication. Instead of treating noise as an artifact to be removed, the proposed system explicitly incorporates realistic noise patterns—such as slang abbreviations, phonetic spelling variations, vowel omissions, and inconsistent capitalization of named entities—into the training process. This design choice enables the model to effectively handle short, incomplete, and irregular user inputs that are commonly encountered in social media posts and chat-based interactions.

Devanagari text is normalized using Unicode NFC normalization to correctly represent conjunct consonants and half-characters, which is essential for accurate phonetic alignment. Non-Devanagari symbols are removed while retaining meaningful punctuation and numerals. The Romanized text is intentionally left unnormalized to preserve user-generated variability, ensuring that the model is exposed to naturally occurring spelling patterns.

To improve robustness against informal typing behaviour, a noise-aware data augmentation strategy is employed. Controlled phonetic noise is injected into the Romanized text by simulating slang substitutions (e.g., kya → ky, nahi → nhi), phonetic character swaps (e.g., v ↔ w, f ↔ ph), and middle-vowel deletion for longer words, reflecting fast and casual typing practices. Additionally, random capitalization is introduced to emulate named entities and inconsistent user capitalization patterns. Each original sample is paired with a noisy variant, and the combined dataset is deduplicated to prevent overfitting while increasing lexical diversity.

The original and noise-augmented samples are then combined to create a diversified training set that enables the model to learn multiple surface forms of the same phonetic representation rather than memorizing standard spellings. This augmented dataset is used to fine-tune an instruction-tuned Gemma-2B-IT model using parameter-efficient LoRA optimization, ensuring robust learning while maintaining low computational cost. Finally, the model is evaluated on an external noisy benchmark dataset (codebyam/Hinglish-Hindi-Transliteration-Dataset) using Character Error Rate (CER), Word Error Rate (WER), and chrF metrics, with a particular focus on short, vowel-omitted, and acronym-heavy inputs. Overall, by explicitly modeling and embracing noise during training, the proposed system significantly improves transliteration robustness and is better suited for real-world applications such as chatbots, sentiment analysis, and machine translation that rely on informal user-generated text.


---

## DATASET DESCRIPTION

***Codebyam/Hinglish-Hindi-Transliteration Dataset:*** This dataset consists of sentence pairs in which Roman-script Hindi is paired with the corresponding Devanagari-script sentence. It is more suited for transliterations than translations, but can be used to train (or continue training) models predicting Roman-script Hindi (transliterated) from standard Hindi or vice versa. This dataset could be useful in tasks such as typing assistance where the writer begins in one script and ends in another, search engine parsing of mixed-script queries, or NLP models that must normalize informal/user-provided data. It also enhances accessibility for users that prefer or need to use Devanagari script, but are less comfortable typing in the native script. But the dataset has its shortcomings: It’s relatively small, so it may not contain enough variation for sloppy real-world text like informal language, abbreviations and typos. As it covers only transliteration and consists of a set of controlled examples, this may not be representative enough of noisy social-media-style Romanized Hindi-script that might affect the generalization ability of advanced language models.

<p align = "center">
  <img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/748300f3-484d-421f-8857-e3b9e56ea8a7" />
</p>
<p align = "center">
  Fig 3: codebyam/Hinglish-Hindi-Transliteration-Dataset
</p>

***Sk_community/Romanized_hindi dataset:*** The Romanized Hindi Dataset released by sk_community from Hugging Face, a large-scale collection of Romanised (Devanagari → Roman) text pairs constructed by pooling together various dataset such as public datasets, synthetically generated data and rule-based transliteration tools. This design ensures broad coverage of linguistic variation in natural Romanized Hindi.

Designed to train and test Hindi ↔ Roman transliteration models, the corpus enables supervised learning of phonetic mappings, spell/typo variations, and informal writing styles. This collection involved 1.78 million parallel pairs over two languages: namely Hindi (in Devanagari script) and Romanized version of Hindi, as well as its release for the research in the area of Transliteration as a benchmark.

### First 3000 samples

<p align = "center">
  <img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/a5fe7c75-51b4-430d-8dd8-68c118b09a19" />
</p>

<p align = "center">
  Fig 4: sk-community/romanized_hindi dataset 
</p>

---
## METHODOLOGY

<p align = "center">
  <img width="700" height="900" alt="image" src="https://github.com/user-attachments/assets/9f66891a-dc8e-4e99-8361-175228df0b87" />
</p>

<p align = "center">
  Fig 5: Methodology framework for transliteration system
</p>

### BLOCK DIAGRAM DESCRIPTION

The block diagram illustrates the end-to-end pipeline of the proposed noise-aware Roman-to-Devanagari transliteration system. 
The process begins with dataset acquisition from the sk-community/romanized_hindi corpus, where an initial subset of 3000 parallel Roman–Devanagari sentence pairs is selected. This is followed by NFC-based normalization and soft cleaning, where unnecessary symbols are removed while preserving informal spellings, and Devanagari text is normalized to correctly handle half-letters and compound characters. 

To model real-world informal writing, a synthetic phonetic noise injection module is applied, introducing realistic variations such as phonetic swaps (e.g., v ↔ w, ee ↔ i), vowel dropping (e.g., namaste → nmste, baat → bat), and inconsistent capitalization of named entities to simulate proper noun usage. Based on these transformations, noise-aware data augmentation is performed to generate multiple spelling variants guided by the detected noise patterns. The original and augmented samples are then merged, duplicates are removed, and the dataset is expanded to approximately 6000 samples. 

The expanded dataset is subsequently split into training and test sets. For training, inputs are tokenized using an instruction-based prompt format, where loss is masked for prompt tokens so that the model learns only the Devanagari output sequence. The instruction-tuned Gemma model is then fine-tuned using LoRA to ensure parameter-efficient learning. 

Finally, the trained model is evaluated on both the internal dataset and an external benchmark dataset (codebyam/Hinglish-Hindi-Transliteration-Dataset) using standard metrics such as CER, WER, chrF, and word-level precision, recall, and F1-score. The final stage involves testing on clean evaluation datasets to assess generalization and robustness in realistic transliteration scenarios.

### PSEUDOCODE: PHONETIC NOISE INJECTION:

#### FUNCTION inject_phonetic_noise(text, p):

    IF text is not string:
        RETURN text
    words = split(text by spaces)
    augmented_words = empty list
    FOR each word in words:
        low_word = lowercase(word)
        # Slang replacement
        IF low_word in SLANG_MAP AND random() < p:
            low_word = random choice from SLANG_MAP[low_word]
        # Phonetic character swaps
        FOR each char, variants in PHONETIC_SWAPS:
            IF char in low_word AND random() < 0.2:
                low_word = replace first occurrence of char with random variant
        # Middle vowel drop (for longer words)
        IF length(low_word) >= 4 AND random() < 0.15:
            vowels = find all vowels in low_word[1:-1]
            IF vowels exist:
                random_vowel = random choice from vowels
                low_word = replace first random_vowel with empty string
        # Random capitalization noise
        IF random() < 0.2:
            low_word = capitalize(low_word)
        ELSE IF random() < 0.05:
            low_word = uppercase(low_word)
        augmented_words.append(low_word)
    RETURN join(augmented_words by spaces)  
    
This function applies realistic phonetic variations to Romanized Hindi text to create augmented training data, and increases dataset diversity to make the model robust to common typing variations, informal language, and noise in user inputs like social media. It splits text into words; for each word, randomly applies slang mapping (e.g., "hai" → "h ey"), phonetic swaps (e.g., "v" ↔ "w"), vowel drops, and capitalization noise; joins augmented words.


### PSEUDOCODE: CLEANING AND NORMALIZATION: 

#### FUNCTION clean_and_normalize(row):

    IF row["Hindi"] is not string OR row["Transliterated Hindi"] is not string:
        RETURN null
    hindi = NFC_normalize(strip(row["Hindi"]))
    hindi = regex_replace(hindi, devanagari_chars + digits + punctuation, "")
    roman = strip(row["Transliterated Hindi"])
    RETURN pair(roman, hindi)

It ensures consistent Unicode representation (NFC normalization for Devanagari half-letters) and removes invalid rows/punctuation for reliable training. In order to do that, it first checks input types; normalizes Hindi text with NFC and strips punctuation; returns cleaned (roman, hindi) tuple or None.

### PSEUDOCODE: TOKENIZATION

#### FUNCTION tokenize_batch(romans, hindis):

    input_ids = empty list
    labels = empty list
    FOR each roman, hindi in zip(romans, hindis):
        prompt = "Hinglish: " + roman
        full_text = prompt + hindi + eos_token
        tokenized_full = tokenizer(full_text, max_length=128, truncate=true)
        tokenized_prompt = tokenizer(prompt, max_length=128, truncate=true, add_special_tokens=false)
        prompt_length = length(tokenized_prompt.input_ids)
        # Mask prompt tokens in labels
        full_labels = array of -100 for first prompt_length tokens
        full_labels + tokenized_full.input_ids[prompt_length:]
        input_ids.append(tokenized_full.input_ids)
        labels.append(full_labels)
    RETURN (input_ids, labels)Prompt-based Tokenization and Loss Masking:

It converts paired Roman-Hindi data into tokenized input_ids and labels for supervised fine-tuning, and prepares data in the format required by the SFTTrainer, masking prompts so the model learns only to predict Devanagari from Hinglish prompts.


---

## OUTCOME

The proposed Roman-to-Hindi script transliteration system was evaluated to assess its ability to accurately generate Devanagari text from noisy Romanized Hindi inputs commonly observed in social media and informal communication. The evaluation focuses on robustness to informal language, abbreviations, vowel omissions, spelling variations, and non-standard phonetic representations.

### Evaluation on CodeByAm Dataset (SecondaryDataset):

The model was first tested on a held-out subset of the codebyam/Hinglish-Hindi-Transliteration-Dataset, which closely reflects real-world noisy Roman text. Performance was measured using standard character- and word-level metrics.

#### Model Performance

Evaluation focuses on **character-level accuracy**, which is the most suitable metric for transliteration tasks, where minor spelling variations are acceptable.

| Metric | Score |
|------|------|
| CER (↓) | 0.1978 |
| WER (↓) | 0.337 |
| CHRF (↑) | 62.639 |
| Word Precision | 1.00 |
| Word Recall | 0.6258 |
| Word F1 | 0.7699 |

The results indicate overall strong and reliable transliteration performance, particularly in handling noisy and informal Romanized Hindi. The Character Error Rate (CER) of 0.1978 shows that nearly 80% of characters are correctly predicted, which is a solid outcome for real-world, noise-heavy transliteration tasks. The Word Error Rate (WER) of 0.337 suggests that while full-word exact matches are more challenging—especially for short or heavily abbreviated inputs—the model still correctly transliterates a majority of words. A chrF score of 62.639 further confirms good character n-gram overlap between predicted and reference outputs, indicating strong preservation of phonetic structure and word form. The perfect word precision (1.00) demonstrates that when the model predicts a word, it is almost always correct, while the word recall of 0.6258 reflects remaining difficulty in recovering all target words from noisy inputs. The resulting word-level F1 score of 0.7699 represents a well-balanced trade-off between precision and recall. Overall, these metrics suggest that the model 
performs well and is robust for practical use, with remaining errors primarily arising from extreme
abbreviations and aggressive vowel omission rather than systematic transliteration failures.

<p align = "center">
  <img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/00aa36c5-18f5-4961-9bc1-475103764080" />
</p>

<p align = "center">
  Fig 6: Model Testing on codebyam/Hinglish-Hindi-Transliteration-dataset
</p>

### Evaluation on Romanized Hindi Dataset (Primary Dataset)

To assess generalization, the model was additionally evaluated on the sk-community/romanized_hindi dataset, which contains cleaner Roman-to-Hindi mappings with less social-media-specific noise.

| Metric | Score |
|------|------|
| CER (↓) | 0.1694 |
| WER (↓) | 0.3011 |
| CHRF (↑) | 71.91 |
| Word Precision | 1.00 |
| Word Recall | 0.5282 |
| Word F1 | 0.6913 |

This result set reflects strong character-level transliteration quality with some remaining challenges at the word level, which is typical for noisy Romanized Hindi. The CER of 0.1694 is notably low, indicating high character-level accuracy and improved handling of phonetic mappings compared to more error-prone baselines. The WER of 0.3011 shows that a majority of words are still correctly transliterated end-to-end, though exact word matches remain difficult when inputs are short or heavily abbreviated. A chrF score of 71.91 is particularly strong and indicates excellent character n-gram overlap between predicted and reference outputs, confirming that phonetic structure and orthographic form are well preserved. The perfect word precision (1.00) again demonstrates that the model’s predictions are highly reliable when produced, while the word recall of 0.5282 suggests that some target words are still missed under extreme noise conditions. Consequently, the word-level F1 score of 0.6913 reflects a moderate but reasonable balance between precision and recall. Overall, this performance can be considered good, especially at the character and phonetic levels, with remaining errors largely attributable to aggressive vowel dropping and highly informal input patterns rather than fundamental model weaknesses.


<p align = "center">
  <img width="800" height="800" alt="image" src="https://github.com/user-attachments/assets/9bf9237b-6ae5-433b-86bf-8d838a7953a3" />
</p>

<p align = "center">
  Fig 7: Model Testing on sk_community/romanized_hindi dataset
</p>

<p align = "center">
  <img width="700" height="700" alt="image" src="https://github.com/user-attachments/assets/db41837d-59db-4d29-979a-9afed9466a52" />
</p>

<p align = "center">
  Fig 8: Performance Comparison of the Proposed Transliteration Model on Internal and External Datasets
</p>

The bar chart presents a comparative evaluation of the proposed transliteration model on an internal dataset (sk-community/romanized_hindi) and an external benchmark dataset (codebyam/Hinglish-Hindi-Transliteration-Dataset). Overall, the model demonstrates consistent and robust performance across both datasets, indicating good generalization beyond the training distribution. On the internal dataset, the model achieves lower CER (0.1694) and WER (0.3011), reflecting stronger character- and word-level accuracy on data closer to the training domain. The higher chrF score (0.7191) further confirms superior preservation of phonetic structure and character n-gram overlap in this setting. In contrast, the external dataset shows slightly higher CER (0.1978) and WER (0.3370), which is expected due to increased noise and stylistic variation; however, the model still maintains competitive performance. Notably, word recall and word-level F1 score are higher on the external dataset, suggesting that the noise-aware training enables the model to recover more target words under real-world, informal conditions. Overall, this comparison highlights that the proposed approach generalizes well, maintaining high phonetic fidelity and balanced word-level performance even when evaluated on unseen and noisier datasets.

---

## CONCLUSION

In this report, we present a robust Hindi back-transliteration system for Romanized text that explicitly accounts for the noise and variability inherent in real-world informal communication. By incorporating a noise-aware preprocessing and data augmentation strategy, the proposed approach effectively models common Romanized Hindi phenomena such as abbreviations, vowel omission, phonetic spellings, inconsistent capitalization, and elongated or echoed characters—patterns that are typically overlooked by conventional transliteration methods trained on clean data.

The instruction-tuned Gemma model, fine-tuned using parameter-efficient LoRA optimization, demonstrates strong transliteration performance, achieving competitive character- and phoneme-level accuracy on both internal and external evaluation datasets. The observed CER and chrF scores indicate that the model preserves phonetic structure and overall word form while generalizing well to unseen and noisy inputs. Qualitative analysis further confirms the model’s robustness in high-noise scenarios, particularly in handling abbreviated forms and informal spellings that are prevalent in everyday digital communication.

Although challenges remain in cases involving morphologically complex words, matra placement, and confusable phonetic character pairs, the proposed pipeline significantly improves transliteration quality under noisy conditions. Overall, this work demonstrates that combining noise-aware data augmentation with parameter-efficient fine-tuning provides an effective and practical solution for real-world Hindi transliteration. The approach establishes a strong foundation for future research in multilingual and low-resource text normalization and transliteration, especially for user-generated content in informal digital environments.

### LIMITATIONS:

Despite demonstrating strong performance on noisy Romanized Hindi, the proposed transliteration system has several limitations that warrant consideration. First, although noise-aware data augmentation improves robustness to informal spellings, the injected noise patterns are rule-driven and probabilistic, which may not fully capture the diversity and unpredictability of real user typing behaviour across social media. As a result, certain unconventional abbreviations or emerging informal language forms may still lead to incorrect transliterations.

Second, the model continues to exhibit errors in handling morphologically complex words, particularly those involving compound formations and subtle matra placement
Third, the evaluation datasets, although external and unseen during training, are still dataset-specific and limited in scale, and may not fully represent the linguistic diversity found in real-time social media streams. 

Finally, the current implementation focuses on sentence-level transliteration and does not incorporate downstream task feedback (e.g., from machine translation or sentiment analysis), which could further refine transliteration quality in end-to-end applications.



