from PyPDF2 import PdfReader
import streamlit as st
from wordcloud import WordCloud
import plotly.express as px
import pandas as pd

from atividade_extensionista_uninter.stats_extract import (
    POLARITY_TO_LABELS,
    get_frequency_by_entity_type,
    get_polarity_frequency,
    get_text_keywords_frequency,
    get_top_words_plot,
)

st.title("Sumarização de dados literários para suporte na análise de obras literárias em língua portuguesa")

uploaded_file = st.file_uploader("Insira o livro (como documento pdf) a ser analisado")

if uploaded_file is not None:
    reader = PdfReader(uploaded_file)

    st.subheader("Palavras-chaves da obra", divider="rainbow")
    word_frequency = get_text_keywords_frequency(reader)
    frequency_cloud = WordCloud().generate_from_frequencies(word_frequency)
    st.image(frequency_cloud.to_array(), use_container_width="always")
    most_common_word, word_count = sorted(
        word_frequency.items(), key=lambda x: x[1], reverse=True
    )[0]
    st.caption(
        f"A palavra com maior frequência foi {most_common_word.title()} possuindo {word_count} citações."
    )

    st.subheader("Personagens da obra", divider="rainbow")
    entity_frequency = get_frequency_by_entity_type("PER", reader)

    entities_cloud = WordCloud().generate_from_frequencies(entity_frequency)
    st.image(entities_cloud.to_array(), use_container_width="always")
    most_common_character, character_count = sorted(
        entity_frequency.items(), key=lambda x: x[1], reverse=True
    )[0]
    st.caption(
        f"O personagem com mais citações ao longo da obra foi {most_common_character} com {character_count} menções."
    )

    st.subheader("Análise de sentimentos", divider="rainbow")
    # st.caption(f'A análise de sentimentos ou mineração de opinião consiste na análise textual visando mensurar estados afetivos que determinadas palavras evocam.')
    lexico = pd.read_csv("./corpus/lexico_v3.0.txt")

    filtered_lexicon = lexico.loc[~lexico["type"].isin(["emot", "htag"])]
    polarity_frequency, word_polarity_frequency = get_polarity_frequency(
        filtered_lexicon, reader
    )
    polarity_frequency_with_labels = {
        POLARITY_TO_LABELS[polarity_frequency]: polarity_count
        for polarity_frequency, polarity_count in polarity_frequency.items()
    }
    polarity_frequency_df = pd.DataFrame(
        {
            "polaridade": polarity_frequency_with_labels.keys(),
            "contagem": polarity_frequency_with_labels.values(),
        }
    )
    fig = px.pie(
        polarity_frequency_df,
        values="contagem",
        names="polaridade",
        title="Classes de sentimentos",
    )
    st.plotly_chart(fig, use_container_width=True)

    for polarity, word_frequency_mapping in word_polarity_frequency.items():
        fig = get_top_words_plot(word_frequency_mapping, POLARITY_TO_LABELS[polarity])
        st.plotly_chart(fig, use_container_width=True)
