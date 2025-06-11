import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import io
import re
import numpy as np

from pycaret.classification import (
    setup as clf_setup,
    compare_models as clf_compare,
    create_model as clf_create,
    pull as clf_pull,
    plot_model as clf_plot,
)
from pycaret.regression import (
    setup as reg_setup,
    compare_models as reg_compare,
    create_model as reg_create,
    pull as reg_pull,
    plot_model as reg_plot,
)

st.set_page_config(page_title="AutoML z PyCaret", layout="wide")
st.title("🔮 Automatyczne modelowanie z PyCaret")

uploaded_file = st.file_uploader("📂 Wybierz plik CSV", type=["csv"])

def best_separator(text, seps=[",", ";", "\t", "|"]):
    max_cols = 0
    best_sep = ","
    for s in seps:
        try:
            df_try = pd.read_csv(io.StringIO(text), sep=s, nrows=5)
            if df_try.shape[1] > max_cols:
                max_cols = df_try.shape[1]
                best_sep = s
        except Exception:
            continue
    return best_sep

def is_time_format(series):
    """Sprawdza, czy większość wartości wygląda na format HH:MM:SS"""
    pattern = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")
    matches = series.dropna().astype(str).apply(lambda x: bool(pattern.match(x)))
    return matches.mean() > 0.8

def convert_time_to_seconds(series):
    def to_sec(t):
        try:
            parts = t.split(':')
            if len(parts) != 3:
                return np.nan
            h, m, s = parts
            return int(h)*3600 + int(m)*60 + int(s)
        except:
            return np.nan
    return series.apply(to_sec)

if uploaded_file is not None:
    try:
        # Wczytujemy plik jako tekst
        file_text = uploaded_file.getvalue().decode("utf-8")

        # Próbujemy wykryć separator Snifferem
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(file_text[:2048])
            sep = dialect.delimiter
        except Exception:
            sep = None

        # Jeśli wykryty separator jest dziwny lub brak, wybierz najlepszy heurystycznie
        if sep is None or sep not in [",", ";", "\t", "|"]:
            sep = best_separator(file_text)

        data_io = io.StringIO(file_text)
        df = pd.read_csv(data_io, sep=sep)
        st.success(f"✅ Plik został wczytany poprawnie! (Separator: '{sep}')")

        st.subheader("📊 Przykładowe dane:")
        st.dataframe(df.head())

        target_column = st.selectbox("🎯 Wybierz kolumnę docelową:", df.columns)

        if target_column:
            # Automatyczna konwersja czasu HH:MM:SS na sekundy
            if is_time_format(df[target_column]):
                st.info(f"🕒 Kolumna '{target_column}' wygląda na czas w formacie HH:MM:SS. Konwertuję na sekundy.")
                df[target_column + "_sec"] = convert_time_to_seconds(df[target_column])
                target_column = target_column + "_sec"

            st.write(f"✅ Wybrana kolumna docelowa to: **{target_column}**")

            # Usuń brakujące wartości z targetu
            missing_target_rows = df[target_column].isnull().sum()
            if missing_target_rows > 0:
                st.warning(f"⚠️ Usunięto {missing_target_rows} wierszy z brakującą wartością w kolumnie docelowej.")
                df = df.dropna(subset=[target_column])

            # Pokaz braki danych
            st.subheader("📉 Liczba brakujących wartości w pozostałych kolumnach:")
            missing_info = df.isnull().sum()
            missing_info = missing_info[missing_info > 0]
            if not missing_info.empty:
                st.dataframe(missing_info)
            else:
                st.markdown("✅ Brak brakujących danych poza kolumną docelową.")

            # Heurystyka: klasyfikacja vs regresja
            unique_values = df[target_column].nunique()
            is_numeric = pd.api.types.is_numeric_dtype(df[target_column])

            if not is_numeric or unique_values <= 20:
                task_type = "classification"
            else:
                task_type = "regression"

            st.info(f"🔍 Sugerowany typ zadania: **{task_type.capitalize()}**")

            run_model = st.button("🚀 Uruchom AutoML")

            if run_model:
                with st.spinner("⏳ Trwa przetwarzanie danych i trenowanie modelu..."):

                    if task_type == "classification":
                        clf_setup(
                            data=df,
                            target=target_column,
                            fix_imbalance=True,
                            numeric_imputation="median",
                            categorical_imputation="mode",
                            verbose=False,
                            session_id=123,
                            use_gpu=False,
                            fold=5,
                        )
                        best_model = clf_compare()
                        results = clf_pull()
                    else:
                        reg_setup(
                            data=df,
                            target=target_column,
                            numeric_imputation="median",
                            categorical_imputation="mode",
                            verbose=False,
                            session_id=123,
                            use_gpu=False,
                            fold=5,
                        )
                        best_model = reg_compare()
                        results = reg_pull()

                st.success("✅ Model został wytrenowany!")

                st.subheader("📈 Porównanie modeli:")
                st.dataframe(results)

                st.subheader("🤖 Najlepszy model:")
                st.write(best_model)

                # Wykres ważności zmiennych
                st.subheader("📊 Ważność zmiennych (Feature Importance):")

                plot_path = "Feature Importance.png"
                if os.path.exists(plot_path):
                    os.remove(plot_path)

                try:
                    if task_type == "classification":
                        clf_plot(best_model, plot="feature", save=True)
                    else:
                        reg_plot(best_model, plot="feature", save=True)

                    if os.path.exists(plot_path):
                        st.image(plot_path, caption="Feature Importance", use_container_width=True)
                    else:
                        raise FileNotFoundError("Wykres nie został wygenerowany.")

                except Exception as e:
                    st.warning(f"⚠️ Nie można wygenerować wykresu ważności dla tego modelu: {e}")
                    st.markdown("Spróbuję użyć modelu Random Forest do wygenerowania wykresu...")

                    try:
                        if task_type == "classification":
                            rf_model = clf_create("rf")
                            clf_plot(rf_model, plot="feature", save=True)
                        else:
                            rf_model = reg_create("rf")
                            reg_plot(rf_model, plot="feature", save=True)

                        if os.path.exists(plot_path):
                            st.image(plot_path, caption="Feature Importance (RandomForest)", use_container_width=True)
                        else:
                            st.error("❌ Nie udało się wygenerować wykresu nawet przy użyciu Random Forest.")
                    except Exception as e2:
                        st.error(f"❌ Błąd przy używaniu Random Forest: {e2}")

    except Exception as e:
        st.error(f"❌ Błąd przy przetwarzaniu: {e}")
else:
    st.info("📂 Proszę załadować plik CSV.")
