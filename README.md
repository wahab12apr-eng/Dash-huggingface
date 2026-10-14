# AI Text Summarizer

A Streamlit web app that summarizes text using Hugging Face Inference models.

## Setup

```
pip install -r requirements.txt
```

Create a `.env` file with your [Hugging Face token](https://huggingface.co/settings/tokens):

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

## Run

```
streamlit run app.py
```

Open http://127.0.0.1:8053 (set in `.streamlit/config.toml`). The page reloads automatically when you save code changes.

When deploying to Streamlit Community Cloud, add `HF_TOKEN` under the app's Secrets instead of using `.env`.

## Features

- Choice of summarization models (BART, DistilBART, Pegasus)
- Adjustable summary length
- Long texts are split into chunks and summarized recursively
- Word counts, compression ratio and timing
- Copy or download the summary; repeated requests are cached
