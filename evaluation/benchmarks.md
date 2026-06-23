# Nusantara LLM Evaluation Benchmarks

## Core Benchmarks

### IndoMMLU
- **Source:** indonlp/indommlu
- **Description:** Indonesian Multi-task Language Understanding
- **Domains:** STEM, humanities, social sciences, medicine, law, business
- **Format:** Multiple-choice (4 options)
- **Metric:** Accuracy
- **Languages:** Indonesian
- **Target:** >60%

### IndoNLG
- **Source:** indonlp/indonlg
- **Description:** Indonesian Natural Language Generation
- **Tasks:** Text summarization, machine translation, paraphrasing
- **Format:** Source text → target text
- **Metric:** ROUGE-1, ROUGE-2, ROUGE-L, BLEU
- **Languages:** Indonesian, English
- **Target:** ROUGE-L >0.35

### NusaX
- **Source:** indonlp/nusax
- **Description:** Regional Language Understanding
- **Tasks:** Sentiment analysis, translation for 12 regional languages
- **Languages:** Acehnese, Balinese, Banjarese, Buginese, Javanese, Madurese, Minangkabau, Palembang, Sundanese
- **Metric:** Accuracy (sentiment), BLEU (translation)
- **Target:** >65% accuracy on regional sentiment

### Indo4B
- **Source:** indonlp/indo4b
- **Description:** Four Indonesian benchmarks in one
- **Tasks:** Natural Language Inference (NLI), Question Answering, Sentiment Analysis, POS Tagging
- **Format:** Task-specific
- **Metric:** Accuracy per task
- **Languages:** Indonesian
- **Target:** >70% average across tasks

## Per-Language Perplexity Targets

| Language | Target PPL | Notes |
|----------|-----------|-------|
| Indonesian | <8 | Native-level fluency |
| Javanese | <12 | Most widely spoken regional language |
| Sundanese | <15 | Second most widely spoken |
| Minangkabau | <18 | Significant speaker base |
| Balinese | <20 | Culturally significant |
| English | <10 | Maintain baseline capabilities |

## Evaluation Protocol

1. Load model in eval mode with BF16 + Flash Attention 2
2. Run each benchmark with fixed random seed (42)
3. For generation tasks, use greedy decoding (do_sample=False)
4. Report mean and standard deviation across 3 runs where applicable
