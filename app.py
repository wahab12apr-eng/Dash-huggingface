import time

import streamlit as st

from huggingface_api import MODELS, DEFAULT_MODEL, SummarizerError, summarize_text


st.set_page_config(page_title="AI Text Summarizer", page_icon="📝", layout="wide")


def word_count(text):
    return len(text.split()) if text else 0


@st.cache_data(show_spinner=False)
def cached_summary(text, model, min_length, max_length):
    # Errors are raised, not cached, so a failed request can be retried.
    return summarize_text(text, model=model, min_length=min_length, max_length=max_length)


def clear():
    st.session_state.input_text = ""
    st.session_state.pop("result", None)


st.title("AI Text Summarizer")
st.caption("Paste an article or paragraph and get a concise summary powered by Hugging Face models.")

with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "Model",
        options=list(MODELS),
        index=list(MODELS).index(DEFAULT_MODEL),
        format_func=MODELS.get,
    )
    min_length, max_length = st.slider(
        "Summary length (tokens)", min_value=10, max_value=300, value=(30, 130), step=10
    )

text = st.text_area("Your text", key="input_text", height=300, placeholder="Enter your text here...")
st.caption(f"{word_count(text)} words · {len(text)} characters")

col1, col2, _ = st.columns([1, 1, 8])
summarize = col1.button("Summarize", type="primary", width="stretch")
col2.button("Clear", on_click=clear, width="stretch")

if summarize:
    if not text.strip():
        st.warning("Please enter some text to summarize.")
    else:
        start = time.perf_counter()
        try:
            with st.spinner("Summarizing..."):
                summary = cached_summary(text, model, min_length, max_length)
        except SummarizerError as e:
            st.error(str(e))
            st.session_state.pop("result", None)
        else:
            st.session_state.result = {
                "summary": summary,
                "in_words": word_count(text),
                "elapsed": time.perf_counter() - start,
                "model": model,
            }

st.divider()
st.subheader("AI Summary")

result = st.session_state.get("result")
if result:
    out_words = word_count(result["summary"])
    reduction = 100 - round(out_words / result["in_words"] * 100) if result["in_words"] else 0

    # st.code provides a built-in copy button.
    st.code(result["summary"], language=None, wrap_lines=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Original", f"{result['in_words']} words")
    m2.metric("Summary", f"{out_words} words")
    m3.metric("Shorter by", f"{reduction}%")
    m4.metric("Time", f"{result['elapsed']:.1f}s")
    st.caption(f"Model: {MODELS[result['model']]}")

    st.download_button("Download summary", result["summary"], file_name="summary.txt")
else:
    st.info("Waiting for input...")
