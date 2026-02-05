import streamlit as st
from engine import load_data, add_internal, predict

st.set_page_config(page_title="FC25 Rush AI", layout="wide")
load_data()

st.title("🤖 FC25 Rush – IA de prédiction")

st.header("➕ Ajouter des scores passés")
text = st.text_area("Format : Equipe A 3-2 Equipe B", height=150)
import re

pattern = re.compile(r"(.+?)\s(\d+)-(\d+)\s(.+)")

if st.button("Ajouter & entraîner"):
    for line in text.splitlines():
        m = pattern.match(line)
        if not m:
            st.warning(f"Ignoré : {line}")
            continue
        a, ga, gb, b = m.groups()
        add_internal(a.strip(), b.strip(), int(ga), int(gb), "")
        st.success(f"Ajouté : {line}")
if st.button("Ajouter & entraîner"):
    for line in text.splitlines():
        try:
            a, rest = line.split(" ", 1)
            score, b = rest.split(" ", 1)
            ga, gb = score.split("-")
            add_internal(a, b, int(ga), int(gb), "")
            st.success(f"Ajouté : {line}")
        except:
            st.warning(f"Ignoré : {line}")

st.header("🔮 Prédiction")
team_a = st.text_input("Equipe A")
team_b = st.text_input("Equipe B")

if st.button("Prédire"):
    res = predict(team_a, team_b)
    if "alert" in res:
        st.error(res["alert"])
    else:
        st.subheader("Top 5 scores probables")
        st.write(res["top_scores"])
        st.write("Confiance :", res["confidence"])
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def export_pdf(result, team1, team2):
    c = canvas.Canvas("prediction.pdf", pagesize=A4)
    c.drawString(50, 800, f"Match : {team1} vs {team2}")
    c.drawString(50, 770, f"Gagnant probable : {result['winner']}")
    c.drawString(50, 740, f"Buts estimés : {result['expected_goals']}")

    y = 710
    for s, p in result["top_scores"]:
        c.drawString(50, y, f"{s} : {p}%")
        y -= 20

    c.save()
