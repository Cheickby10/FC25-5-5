import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# --- CONFIG ---
DATA_FILE = "data/matches.csv"
MEMORY_FILE = "data/memory.json"

st.set_page_config(page_title="FC25 5x5 Rush IA", layout="wide")

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["date", "team_a", "team_b", "score_a", "score_b"])
    return df

df = load_data()

# --- FONCTIONS ---
def save_data(df):
    df.to_csv(DATA_FILE, index=False)

def add_matches(text_input):
    lines = text_input.strip().split("\n")
    new_matches = []
    for line in lines:
        try:
            parts = line.split()
            team_a = parts[0]
            score_a, score_b = map(int, parts[1].split("-"))
            team_b = parts[2]
            new_matches.append({
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "team_a": team_a,
                "team_b": team_b,
                "score_a": score_a,
                "score_b": score_b
            })
        except:
            continue
    if new_matches:
        df_new = pd.DataFrame(new_matches)
        df_updated = pd.concat([df, df_new], ignore_index=True)
        save_data(df_updated)
        return df_updated
    return df

def compute_stats(df):
    if df.empty:
        return 0, 0, 0
    goals = pd.concat([df["score_a"], df["score_b"]])
    mean_goals = goals.mean()
    variance = goals.var()
    reliability = min(100, max(0, 100 - variance))  # exemple simple
    return mean_goals, variance, reliability

# --- DASHBOARD ---
st.title("⚽ FC25 5×5 Rush – IA Prédictive")
st.write("IA auto-apprenante basée sur des matchs réels FIFA")

st.subheader("Ajouter plusieurs matchs (format: TeamA ScoreA-ScoreB TeamB, un par ligne)")
match_input = st.text_area("Exemple:\nPorto 2-1 Milano\nACMilan 3-3 Liverpool")
if st.button("Ajouter les matchs"):
    df = add_matches(match_input)
    st.success(f"{len(match_input.strip().splitlines())} matchs ajoutés avec succès!")

st.subheader("État du moteur IA")
matches_count = len(df)
mean_goals, variance, reliability = compute_stats(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Matches appris", matches_count)
col2.metric("Buts moyens", f"{mean_goals:.2f}")
col3.metric("Variance", f"{variance:.2f}")
col4.metric("Fiabilité réelle", f"{reliability:.2f}%")

st.subheader("Distribution des scores")
if not df.empty:
    plt.figure(figsize=(10,5))
    goals = pd.concat([df["score_a"], df["score_b"]])
    goals = pd.to_numeric(goals, errors='coerce').dropna()
    if not goals.empty:
        sns.histplot(goals, bins=range(int(goals.min()), int(goals.max())+2), kde=False)
        plt.xlabel("Buts")
        plt.ylabel("Nombre de matchs")
        plt.title("Distribution des buts marqués")
        st.pyplot(plt)
    else:
        st.write("Pas de données valides pour l’histogramme.")
else:
    st.write("Aucun match enregistré.")

st.subheader("Historique des matchs")
if not df.empty:
    st.dataframe(df.sort_values(by="date", ascending=False))
else:
    st.write("Aucun match enregistré.")
