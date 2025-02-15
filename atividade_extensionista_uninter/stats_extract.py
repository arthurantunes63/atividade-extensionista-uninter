from collections import Counter

import streamlit as st
import spacy
from spacy.lang.pt.stop_words import STOP_WORDS


nlp = spacy.load("pt_core_news_lg")

option_to_label = {
    "Pessoas": "PER",
    "Entidade Geopolítica (GPE)": "GPE",
    "Localidade": "LOC",
}
st.title("Análise de obras literárias - Extração de valor de dados textuais")

POLARITY_TO_LABELS = {
    0: "neutro",
    1: "positivo",
    -1: "negativo",
}


def get_polarity_frequency(filtered_lexicon, reader):
    polarity_frequency = {
        0: 0,
        1: 0,
        -1: 0,
    }
    word_polarity_frequency = {0: {}, 1: {}, -1: {}}
    for page_number, page in enumerate(reader.pages[1:]):
        text = page.extract_text()
        doc = nlp(text)
        for token in doc:
            if not token.pos_ == "VERB" and not token.is_stop:
                if token.text.lower() in filtered_lexicon["term"].values:
                    polarity = filtered_lexicon[filtered_lexicon["term"] == token.text][
                        "polarity"
                    ]
                    if polarity.any():
                        polarity = polarity.iloc[0]
                        polarity_frequency[polarity] += 1

                        if token.text.lower() in word_polarity_frequency[polarity]:
                            word_polarity_frequency[polarity][token.text.lower()] += 1
                        else:
                            word_polarity_frequency[polarity][token.text.lower()] = 1
    return polarity_frequency, word_polarity_frequency


def get_text_keywords_frequency(reader):
    word_frequency = Counter()
    for page_number, page in enumerate(reader.pages[1:]):
        text = page.extract_text()
        doc = nlp(text)
        filtered_tokens = (
            token.lemma_.replace("\n", "")
            for token in doc
            if token.lemma_ not in STOP_WORDS
            and not token.is_punct
            and not token.is_digit
            and not token.is_space
            and not token.is_stop
            and not token.pos_ == "PROPN"
            and not token.pos_ == "VERB"
        )
        word_frequency.update(filtered_tokens)
    return word_frequency


def get_frequency_by_entity_type(entity_label: str, reader):
    entity_frequency = Counter()
    for page_number, page in enumerate(reader.pages[1:]):
        text = page.extract_text()
        doc = nlp(text)

        person_entities = (
            ent.text.replace("\n", "") for ent in doc.ents if ent.label_ == entity_label
        )
        entity_frequency.update(person_entities)
    return entity_frequency


COLOR_BY_POLARITY = {"positivo": "yellow", "negativo": "blue", "neutro": "gray"}


def get_top_words_plot(word_frequency_dict, polarity):
    # Sort the dictionary by values in descending order
    sorted_word_freq = sorted(
        word_frequency_dict.items(), key=lambda x: x[1], reverse=True
    )

    # Extract top 10 words and their frequencies
    top_words = [item[0] for item in sorted_word_freq[:10]]
    frequencies = [item[1] for item in sorted_word_freq[:10]]

    # Create a bar chart using Plotly Express
    fig = px.bar(
        x=top_words,
        y=frequencies,
        labels={"x": "Palavras", "y": "Contagem"},
        title=f"Top 10 palavras por frequência da polaridade {polarity}",
        color_discrete_sequence=[COLOR_BY_POLARITY[polarity]] * len(top_words),
    )

    return fig
