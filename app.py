import streamlit as st
import pandas as pd
import re
from pathlib import Path
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from engine import IAEngine
from datetime import datetime

# -----------------------------
# Configuration
# -----------------------------
st.set_page_config(page_title="FC25 5x5 Rush - IA", layout="wide")
DATA_PATH = Path("data/matches.csv")
st.title("⚽ FC25 5×5 Rush – IA Prédictive")

# -----------------------------
# Section 1 : Ajouter plusieurs matchs
# -----------------------------
st.header("📥 Ajouter plusieurs matchs")
st.write("Format par ligne : `EquipeA scoreA-scoreB EquipeB`")

texte_matchs = st.text_area("Collez vos matchs ici (un match par ligne)", height=200)

if st.button("Ajouter les matchs au bot"):
    if texte_matchs:
        lignes = texte_matchs.strip().split("\n")
        nouvelles_lignes = []
        for ligne in lignes:
            match = re.match(r"(.+?) (\d+)-(\d+) (.+)", ligne.strip())
            if match:
                nouvelles_lignes.append({
                    "team_a": match.group(1).strip(),
                    "team_b": match.group(4).strip(),
                    "ga": int(match.group(2)),
                    "gb": int(match.group(3)),
                    "date": datetime.today().strftime("%Y-%m-%d")
                })
            else:
                st.warning(f"Ligne invalide ignorée : {ligne}")

        if nouvelles_lignes:
            if DATA_PATH.exists():
                df = pd.read_csv(DATA_PATH)
            else:
                df = pd.DataFrame(columns=["team_a","team_b","ga","gb","date"])
            df = pd.concat([df, pd.DataFrame(nouvelles_lignes)], ignore_index=True)
            df.to_csv(DATA_PATH, index=False)
            st.success(f"{len(nouvelles_lignes)} matchs ajoutés et sauvegardés !")
        else:
            st.error("Aucun match valide à ajouter.")
    else:
        st.warning("Veuillez coller vos matchs avant.")

# -----------------------------
# Section 2 : Dashboard
# -----------------------------
st.header("📊 Dashboard des performances")

if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
    if not df.empty:
        nb_matchs = len(df)
        buts_moyens = (df["ga"].sum() + df["gb"].sum()) / nb_matchs
        variance = df[["ga","gb"]].stack().var()
        fiabilite = max(0, 100 - variance*10)

        # Stats clés
        col1, col2, col3 = st.columns(3)
        col1.metric("Matchs appris", nb_matchs)
        col2.metric("Buts moyens par match", f"{buts_moyens:.2f}")
        col3.metric("Fiabilité réelle (%)", f"{fiabilite:.2f}")

        # Graphique distribution des scores
        st.subheader("Distribution des scores")
        plt.figure(figsize=(10,4))
        plt.hist(df["ga"], bins=range(0,max(df[["ga","gb"]].max()+2,7)), color="blue", alpha=0.6, label="Team A")
        plt.hist(df["gb"], bins=range(0,max(df[["ga","gb"]].max()+2,7)), color="red", alpha=0.6, label="Team B")
        plt.xlabel("Buts")
        plt.ylabel("Nombre de fois")
        plt.legend()
        st.pyplot(plt.gcf())

        # Derniers matchs ajoutés
        st.subheader("Derniers matchs")
        st.dataframe(df.tail(10))
    else:
        st.info("Le fichier CSV est vide, ajoutez des matchs pour générer le dashboard.")
else:
    st.info("Aucun fichier CSV trouvé. Ajoutez des matchs pour commencer.")

# -----------------------------
# Section 3 : Prédiction IA
# -----------------------------
st.header("🤖 Prédictions IA")

team_a = st.text_input("Équipe A pour la prédiction")
team_b = st.text_input("Équipe B pour la prédiction")

if st.button("Prédire le match") and team_a and team_b:
    if DATA_PATH.exists() and len(pd.read_csv(DATA_PATH))>0:
        ia = IAEngine(DATA_PATH)
        ia.train()
        score_exp, confidence, top5, incest = ia.predict(team_a, team_b)

        st.subheader("⚡ Résultat attendu")
        st.write(f"**Score attendu :** {score_exp[0]} - {score_exp[1]}")
        st.write(f"**Fiabilité :** {confidence*100:.1f}%")
        if incest:
            st.warning("⚠️ Ce match a été joué trop souvent récemment (incestueux)")

        st.write("**Top 5 scores les plus probables :**")
        for s,p in top5:
            st.write(f"{s[0]}-{s[1]} ({p*100:.1f}%)")

        # Export PDF
        def export_pdf(result, t1, t2):
            c = canvas.Canvas(f"prediction_{t1}_{t2}.pdf")
            c.drawString(50,800,f"Match : {t1} vs {t2}")
            c.drawString(50,770,f"Score attendu : {result[0][0]} - {result[0][1]}")
            c.drawString(50,740,f"Fiabilité : {result[1]*100:.1f}%")
            if result[3]:
                c.drawString(50,710,"⚠️ Match incestueux détecté")
            y = 680
            c.drawString(50,y,"Top 5 scores probables :")
            y -= 20
            for s,p in result[2]:
                c.drawString(50,y,f"{s[0]}-{s[1]} ({p*100:.1f}%)")
                y -= 20
            c.save()

        if st.button("📄 Exporter la prédiction en PDF"):
            export_pdf((score_exp, confidence, top5, incest), team_a, team_b)
            st.success("PDF généré avec succès !")
    else:
        st.warning("Pas assez de données pour prédire ce match.")
