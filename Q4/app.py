import re
import sys
import time
import random
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import Q4Config
from src.model_loader import load_all
from src.ngram_q4 import train_shared_models
from src.live_pipeline import LiveEditorPipeline
from src.passage_sampler import sample_passage
from src.pcfg_parser import PCFGParser
from src.final_analysis import FinalAnalyzer
from src.benchmark import run_q4_benchmark


st.set_page_config(page_title='Q4 Integrated Background Editor', layout='wide')


@st.cache_resource
def load_runtime():
    config = Q4Config()
    models = load_all()
    shared_lm = train_shared_models(models['q1_lm'], models['q3_data']['sentences'], k=config.q4_add_k)
    pcfg = PCFGParser().train()
    return models, shared_lm, pcfg


try:
    models, shared, pcfg = load_runtime()
except Exception as exc:
    st.error(str(exc))
    st.stop()


cfg = Q4Config()
st.title('Question 4 — Integrated Background Editor')
st.caption('Q1 joint segmentation/POS + Q3 spelling + Q4 PCFG/n-gram grammar analysis')


def alert_text(alert):
    return f"[{alert.kind}] {alert.message} — {alert.latency_ms:.2f} ms"


def render_alert(alert):
    if alert.kind == 'SEGMENT-ALERT':
        st.warning(alert_text(alert))
    elif alert.kind == 'SPELL-ALERT':
        st.error(alert_text(alert))
    else:
        st.info(alert_text(alert))


def sentence_groups_from_history(text, pipe):
    """Build final sentence groups from raw-token history.

    Unlike a simple word-count split, this remains correct when one raw input
    token is segmented into two or more final tokens by Q1.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    history = pipe.history
    if not history:
        return [], {}, {}

    # First determine the final-token span contributed by each raw token. A Q1
    # merge changes token count, but a Q3 real-word correction is one-for-one.
    raw_groups = []
    cursor = 0
    for si, line in enumerate(lines):
        raw_count = len(line.split())
        raw_groups.append(history[cursor:cursor + raw_count])
        cursor += raw_count
    if cursor < len(history):
        raw_groups.append(history[cursor:])

    groups = []
    merge_counts = {}
    spell_counts = {}
    final_cursor = 0
    for si, segment in enumerate(raw_groups):
        count = sum(len(item.final_tokens) for item in segment)
        # IMPORTANT: read from pipe.tokens, not item.final_tokens. Trigger-level
        # real-word correction mutates the shared final stream after a token has
        # already been recorded in history.
        groups.append(pipe.tokens[final_cursor:final_cursor + count])
        final_cursor += count
        merge_counts[si] = sum(int(item.segmentation_merge) for item in segment)
        spell_counts[si] = sum(
            1 for item in segment
            for alert in item.alerts if alert.kind == 'SPELL-ALERT'
        )

    # Trigger-level real-word fixes are already reflected in pipe.tokens. Count
    # them for the sentence containing the corrected absolute token.
    cumulative = []
    total = 0
    for group in groups:
        total += len(group)
        cumulative.append(total)
    for absolute, old, new in pipe.realword_corrections:
        for si, end in enumerate(cumulative):
            if absolute < end:
                spell_counts[si] = spell_counts.get(si, 0) + 1
                break

    return [group for group in groups if group], merge_counts, spell_counts


mode = st.radio('Mode', ['Simulated typing', 'Live typing'], horizontal=True)

if mode == 'Simulated typing':
    seed = st.number_input('Random seed (0 = random)', min_value=0, value=42, step=1)
    speed = st.slider('Seconds between simulated words', 0.0, 0.5, 0.08, 0.01)

    if st.button('Run random 5–8 sentence passage'):
        raw = sample_passage(None if seed == 0 else int(seed))
        pipe = LiveEditorPipeline(models['q1_decoder'], models['q3_corrector'], shared, cfg)
        live = st.empty()
        alerts_box = st.container()
        all_sentences = []
        merge_counts = {}
        spell_counts = {}
        rng = random.Random(None if seed == 0 else int(seed))

        for si, sentence in enumerate(raw):
            words = [w.lower() for w in sentence if w.isalpha()]
            i = 0
            output = []
            merge_count = 0
            spell_count = 0
            while i < len(words):
                if i < len(words) - 1 and rng.random() < cfg.merge_probability:
                    token = words[i] + words[i + 1]
                    i += 2
                else:
                    token = words[i]
                    i += 1

                item = pipe.process_token(token)
                output.extend(item.final_tokens)
                merge_count += int(item.segmentation_merge)
                spell_count += sum(1 for alert in item.alerts if alert.kind == 'SPELL-ALERT')

                with live.container():
                    st.markdown('**Live text:** ' + ' '.join(pipe.tokens))
                with alerts_box:
                    for alert in item.alerts:
                        render_alert(alert)
                if speed:
                    time.sleep(speed)

            all_sentences.append(output)
            merge_counts[si] = merge_count
            spell_counts[si] = spell_count

        # Trigger-level real-word corrections may change earlier finalized words.
        # Rebuild sentence snapshots from the final shared stream so final analysis
        # never uses stale pre-correction tokens.
        sentence_token_counts = [len(sent) for sent in all_sentences]
        all_sentences = []
        cursor = 0
        for count in sentence_token_counts:
            all_sentences.append(pipe.tokens[cursor:cursor + count])
            cursor += count
        cumulative = []
        total = 0
        for sent in all_sentences:
            total += len(sent)
            cumulative.append(total)
        for absolute, old, new in pipe.realword_corrections:
            for si, end in enumerate(cumulative):
                if absolute < end:
                    spell_counts[si] = spell_counts.get(si, 0) + 1
                    break

        analyzer = FinalAnalyzer(models['q1_decoder'], models['q1_tagger'], shared, pcfg)
        df = analyzer.analyze(all_sentences, merge_counts, spell_counts)
        st.subheader('Final sentence analysis')
        st.dataframe(df, use_container_width=True)
        st.subheader('Latency')
        st.json(pipe.averages())

else:
    st.write(
        'Type normally. A token is processed as soon as a space/newline completes it. '
        'The final unfinished token is held until you type a delimiter. '
        'For clean sentence-level final analysis, use one sentence per line.'
    )

    if 'live_pipe' not in st.session_state:
        st.session_state.live_pipe = LiveEditorPipeline(models['q1_decoder'], models['q3_corrector'], shared, cfg)
        st.session_state.live_completed_tokens = []
        st.session_state.live_alerts = []
        st.session_state.live_text = ''

    text = st.text_area(
        'Live text',
        height=180,
        placeholder='Type a sentence and keep typing…',
        key='live_editor_text',
    )

    # Only complete whitespace-delimited tokens enter the pipeline. This is
    # genuine incremental processing across Streamlit reruns: already-processed
    # tokens are stored in session_state and are not processed again.
    completed = text.split()
    if text and not text.endswith((' ', '\n', '\t')) and completed:
        completed = completed[:-1]

    old_completed = st.session_state.live_completed_tokens
    if completed[:len(old_completed)] != old_completed:
        # The user edited/deleted already-processed text. Rebuild deterministically
        # from the current completed prefix rather than mixing old state with new.
        st.session_state.live_pipe.reset()
        st.session_state.live_completed_tokens = []
        st.session_state.live_alerts = []
        old_completed = []

    new_tokens = completed[len(old_completed):]
    for token in new_tokens:
        item = st.session_state.live_pipe.process_token(token)
        st.session_state.live_completed_tokens.append(token)
        st.session_state.live_alerts.extend(item.alerts)

    st.subheader('Live stream')
    st.markdown('**Processed text:** ' + ' '.join(st.session_state.live_pipe.tokens))

    if st.session_state.live_alerts:
        st.subheader('Alerts')
        for alert in st.session_state.live_alerts:
            render_alert(alert)
    else:
        st.caption('No alerts yet. Keep typing; segmentation/spelling runs per completed token and grammar runs every N words.')

    st.caption(f"Grammar triggers executed: {len(st.session_state.live_pipe.trigger_latencies)}")
    st.subheader('Live latency')
    st.json(st.session_state.live_pipe.averages())

    if st.button('Run final analysis on processed text'):
        sentences, merge_counts, spell_counts = sentence_groups_from_history(text, st.session_state.live_pipe)
        if not sentences:
            st.warning('No completed tokens are available yet.')
        else:
            df = FinalAnalyzer(models['q1_decoder'], models['q1_tagger'], shared, pcfg).analyze(sentences, merge_counts, spell_counts)
            st.subheader('Final sentence analysis')
            st.dataframe(df, use_container_width=True)

st.divider()
st.subheader('Q4 Speed Demon')
st.caption('Exactly 1,000 simulated words; compares the full Q1/Q3 per-token layer with the isolated grammar/real-word trigger layer.')

if st.button('Benchmark exactly 1,000 simulated words'):
    with st.spinner('Running benchmark…'):
        benchmark = run_q4_benchmark(models['q1_decoder'], models['q3_corrector'], shared, cfg)
    st.json(benchmark)
    if benchmark['full_pipeline_seconds'] > 10:
        st.warning(
            'The full 1,000-word layer is expensive on this machine. This is a useful result: '
            'the report should discuss throttling/optimizing OOV segmentation rather than hiding the measured overhead.'
        )
