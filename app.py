import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from engine import (
    add_match,
    predict_match,
    confidence_level
)

st.set_page_config(page_title="FC25 5x5 Rush IA", layout="wide")

st.title("⚽ FC25 5×5 Rush – IA Prédictive")
st.caption("IA auto-apprenante basée sur des matchs réels FIFA")

# =========================
# Chargement données
# =========================
DATA_PATH = "data/matches.csv"

try:
    df = pd.read_csv(DATA_PATH)
except:
    df = pd.DataFrame(columns=["team_a", "team_b", "ga", "gb", "date"])

# =========================
# DASHBOARD – STATISTIQUES
# =========================
st.subheader("📊 État du moteur IA")

col1, col2, col3, col4 = st.columns(4)

total_matches = len(df)
avg_goals = (df["ga"].sum() + df["gb"].sum()) / max(total_matches, 1)
variance = df["ga"].append(df["gb"]).var() if total_matches > 1 else 0

# Fiabilité réelle : prédiction issue correcte (approximation)
def real_accuracy(df):
    if len(df) < 5:
        return 0
    correct = 0
    tested = 0
    for _, row in df.iterrows():
        pred = predict_match(row["team_a"], row["team_b"])
        if pred is None:
            continue
        winner = (
            row["team_a"] if row["ga"] > row["gb"]
            else row["team_b"] if row["gb"] > row["ga"]
            else "Draw"
        )
        if pred["winner"] == winner:
            correct += 1
        tested += 1
    return round((correct / tested) * 100, 1) if tested > 0 else 0

accuracy = real_accuracy(df)

col1.metric("Matchs appris", total_matches)
col2.metric("Buts moyens", round(avg_goals, 2))
col3.metric("Stabilité (variance)", round(variance, 2))
col4.metric("Fiabilité réelle", f"{accuracy}%")

# =========================
# AJOUT MATCH
# =========================
st.subheader("➕ Ajouter un match (mémoire IA)")

with st.form("add_match"):
    c1, c2, c3, c4 = st.columns(4)
    team_a = c1.text_input("Équipe A")
    team_b = c2.text_input("Équipe B")
    ga = c3.number_input("Buts A", 0, 20, 0)
    gb = c4.number_input("Buts B", 0, 20, 0)
    submitted = st.form_submit_button("Ajouter")

    if submitted and team_a and team_b:
        add_match(team_a, team_b, ga, gb)
        st.success("Match ajouté. IA mise à jour.")

# =========================
# PRÉDICTION
# =========================
st.subheader("🔮 Prédiction de match")

c1, c2 = st.columns(2)
team1 = c1.text_input("Équipe 1")
team2 = c2.text_input("Équipe 2")

if st.button("Prédire"):
    result = predict_match(team1, team2)

    if result is None:
        st.warning("Pas assez de données pour prédire.")
    else:
        conf = confidence_level(result["confidence"])

        st.markdown(f"""
### 🏆 Gagnant probable : **{result['winner']}**
- Buts estimés : **{result['expected_goals']}**
- Niveau de confiance : **{conf}**
        """)

        st.write("### 🎯 Top 5 scores exacts")
        for s, p in result["top_scores"]:
            st.write(f"{s} — {p}%")

# =========================
# DISTRIBUTION DES SCORES
# =========================
st.subheader("📈 Distribution des buts")

if total_matches > 0:
    goals = df["ga"].tolist() + df["gb"].tolist()
    fig, ax = plt.subplots()
    ax.hist(goals, bins=10)
    ax.set_title("Distribution des buts")
    st.pyplot(fig)

# =========================
# HISTORIQUE
# =========================
st.subheader("📜 Historique des matchs")
st.dataframe(df.tail(10), use_container_width=True)
