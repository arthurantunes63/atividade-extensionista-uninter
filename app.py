from collections import Counter

import spacy
from spacy.lang.pt.stop_words import STOP_WORDS

from PyPDF2 import PdfReader
import streamlit as st
from wordcloud import WordCloud
import plotly.express as px
import pandas as pd


nlp = spacy.load("pt_core_news_lg")

option_to_label = {
    "Pessoas": 'PER',
    "Entidade Geopolítica (GPE)": "GPE",
    "Localidade": "LOC"
}
st.title('Análise de obras literárias - Extração de valor de dados textuais')

POLARITY_TO_LABELS = {
    0:"neutro",
    1: "positivo",
    -1: "negativo",
}

def get_polarity_frequency(filtered_lexicon):
    polarity_frequency = {
        0: 0,
        1: 0,
        -1: 0,
    }
    word_polarity_frequency = {
        0: {},
        1: {},
        -1: {}
    }
    for page_number, page in enumerate(reader.pages[1:]):
        text = page.extract_text()
        doc = nlp(text)
        for token in doc:
            if not token.pos_ == "VERB" and not token.is_stop:
                if token.text.lower() in filtered_lexicon['term'].values:
                    polarity = filtered_lexicon[filtered_lexicon['term'] == token.text]['polarity']
                    if polarity.any():
                        polarity = polarity.iloc[0]
                        polarity_frequency[polarity] += 1
                        
                        if token.text.lower() in word_polarity_frequency[polarity]:
                            word_polarity_frequency[polarity][token.text.lower()] += 1
                        else:
                            word_polarity_frequency[polarity][token.text.lower()] = 1
    return polarity_frequency, word_polarity_frequency

def get_text_keywords_frequency():
    word_frequency = Counter()
    for page_number, page in enumerate(reader.pages[1:]):
        text = page.extract_text()
        doc = nlp(text)
        filtered_tokens = (
            token.lemma_.replace('\n', '')
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


def get_frequency_by_entity_type(entity_label: str):
    entity_frequency = Counter()
    for page_number, page in enumerate(reader.pages[1:]):
        text = page.extract_text()
        doc = nlp(text)
        
        person_entities = (
            ent.text.replace('\n', '')
            for ent in doc.ents
            if ent.label_ == entity_label
        )
        entity_frequency.update(person_entities)
    return entity_frequency

COLOR_BY_POLARITY = {
    "positivo": "yellow",
    "negativo": "blue",
    "neutro": "gray"
}
def get_top_words_plot(word_frequency_dict, polarity):
    # Sort the dictionary by values in descending order
    sorted_word_freq = sorted(word_frequency_dict.items(), key=lambda x: x[1], reverse=True)

    # Extract top 10 words and their frequencies
    top_words = [item[0] for item in sorted_word_freq[:10]]
    frequencies = [item[1] for item in sorted_word_freq[:10]]

    # Create a bar chart using Plotly Express
    fig = px.bar(x=top_words, y=frequencies, labels={'x': 'Palavras', 'y': 'Contagem'},
                 title=f'Top 10 palavras por frequência da polaridade {polarity}', color_discrete_sequence =[COLOR_BY_POLARITY[polarity]]*len(top_words))

    return fig


uploaded_file = st.file_uploader("Insira o livro (como documento pdf) a ser analisado")

if uploaded_file is not None:
    reader = PdfReader(uploaded_file)

    st.subheader('Palavras-chaves da obra', divider='rainbow')
    word_frequency = get_text_keywords_frequency()
    frequency_cloud = WordCloud().generate_from_frequencies(word_frequency)
    st.image(frequency_cloud.to_array(), use_column_width="always")
    most_common_word, word_count = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)[0]
    st.caption(f'A palavra com maior frequência foi {most_common_word} possuindo {word_count} citações.')

    st.subheader('Personagens da obra', divider='rainbow')
    entity_frequency = get_frequency_by_entity_type("PER")
    entities_cloud = WordCloud().generate_from_frequencies(entity_frequency)
    st.image(entities_cloud.to_array(), use_column_width="always")
    most_common_character, character_count = sorted(entity_frequency.items(), key=lambda x: x[1], reverse=True)[0]
    st.caption(f'O personagem com mais citações ao longo da obra foi {most_common_character} com {character_count} menções.')

    st.subheader('Análise de sentimentos', divider='rainbow')
    # st.caption(f'A análise de sentimentos ou mineração de opinião consiste na análise textual visando mensurar estados afetivos que determinadas palavras evocam.')
    lexico = pd.read_csv("./corpus/lexico_v3.0.txt")

    filtered_lexicon = lexico.loc[~lexico['type'].isin(["emot", "htag"])]
    polarity_frequency, word_polarity_frequency = get_polarity_frequency(filtered_lexicon)
    polarity_frequency_with_labels = {POLARITY_TO_LABELS[polarity_frequency]: polarity_count for polarity_frequency, polarity_count in polarity_frequency.items()}
    polarity_frequency_df = pd.DataFrame({"polaridade": polarity_frequency_with_labels.keys(), "contagem": polarity_frequency_with_labels.values()})
    fig = px.pie(polarity_frequency_df, values='contagem', names='polaridade', title='Classes de sentimentos')
    st.plotly_chart(fig, use_container_width=True)

    for polarity, word_frequency_mapping in word_polarity_frequency.items():
        fig = get_top_words_plot(word_frequency_mapping, POLARITY_TO_LABELS[polarity])
        st.plotly_chart(fig, use_container_width=True)
    