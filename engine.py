import csv
import os
from datetime import datetime
from collections import defaultdict, Counter
import math

DATA_PATH = "data/matches.csv"


# ---------- MÉMOIRE PERSISTANTE ----------

def load_matches():
    if not os.path.exists(DATA_PATH):
        return []

    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_match(team_a, team_b, ga, gb):
    exists = os.path.exists(DATA_PATH)

    with open(DATA_PATH, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["team_a", "team_b", "ga", "gb", "date"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not exists or os.stat(DATA_PATH).st_size == 0:
            writer.writeheader()

        writer.writerow({
            "team_a": team_a,
            "team_b": team_b,
            "ga": ga,
            "gb": gb,
            "date": datetime.now().isoformat()
        })


# ---------- PONDÉRATION TEMPORELLE ----------

def recency_weight(date_str):
    try:
        d = datetime.fromisoformat(date_str)
        days = (datetime.now() - d).days
        return math.exp(-days / 30)  # décroissance douce
    except:
        return 0.5


# ---------- EXTRACTION DES STATS ----------

def team_stats(matches, team):
    scored = []
    conceded = []

    for m in matches:
        w = recency_weight(m["date"])
        if m["team_a"] == team:
            scored.append(int(m["ga"]) * w)
            conceded.append(int(m["gb"]) * w)
        elif m["team_b"] == team:
            scored.append(int(m["gb"]) * w)
            conceded.append(int(m["ga"]) * w)

    return scored, conceded


# ---------- PRÉDICTION ----------

def predict(team_a, team_b):
    matches = load_matches()

    sa, ca = team_stats(matches, team_a)
    sb, cb = team_stats(matches, team_b)

    n = min(len(sa), len(sb))

    # ----- Détection d'incertitude -----
    if n < 5:
        return {
            "unstable": True,
            "reason": "Données insuffisantes",
            "confidence": 0.25
        }

    avg_a = sum(sa) / len(sa)
    avg_b = sum(sb) / len(sb)

    avg_ca = sum(ca) / len(ca)
    avg_cb = sum(cb) / len(cb)

    # estimation buts
    exp_a = max(0, round((avg_a + avg_cb) / 2))
    exp_b = max(0, round((avg_b + avg_ca) / 2))

    # ----- Distribution des scores -----
    score_dist = Counter()

    for m in matches:
        if {m["team_a"], m["team_b"]} == {team_a, team_b}:
            w = recency_weight(m["date"])
            score = f"{m['ga']}-{m['gb']}" if m["team_a"] == team_a else f"{m['gb']}-{m['ga']}"
            score_dist[score] += w

    if not score_dist:
        # fallback cohérent FIFA 6min
        score_dist = Counter({
            f"{exp_a}-{exp_b}": 1.0,
            f"{exp_a+1}-{exp_b}": 0.6,
            f"{exp_a}-{exp_b+1}": 0.6,
            f"{exp_a+1}-{exp_b+1}": 0.4,
            f"{max(0,exp_a-1)}-{max(0,exp_b-1)}": 0.3
        })

    top_scores = score_dist.most_common(5)

    # ----- Score confidence -----
    variance = abs(avg_a - avg_b) + abs(avg_ca - avg_cb)
    confidence = min(0.9, 0.4 + (n / 20) - (variance / 10))
    confidence = max(0.3, confidence)

    winner = (
        team_a if exp_a > exp_b else
        team_b if exp_b > exp_a else
        "Match nul"
    )

    return {
        "unstable": False,
        "winner": winner,
        "expected_goals": (exp_a, exp_b),
        "top_scores": top_scores,
        "confidence": round(confidence, 2),
        "samples": n
      }
import pandas as pd
import math
from collections import Counter
from datetime import datetime

DATA_PATH = "data/matches.csv"

def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except:
        return pd.DataFrame(columns=["team_a","team_b","ga","gb","date"])

def save_data(df):
    df.to_csv(DATA_PATH, index=False)

# =========================
# AJOUT MATCH + AUTO-UPDATE
# =========================
def add_match(team_a, team_b, ga, gb):
    df = load_data()
    new_row = {
        "team_a": team_a,
        "team_b": team_b,
        "ga": ga,
        "gb": gb,
        "date": datetime.now().isoformat()
    }
    df = pd.concat([df, pd.DataFrame([new_row])])
    save_data(df)

# =========================
# MÉMOIRE PAR ÉQUIPE
# =========================
def team_profile(team, df):
    games = df[(df.team_a == team) | (df.team_b == team)]
    if len(games) == 0:
        return None

    goals_for = []
    goals_against = []

    for _, r in games.iterrows():
        if r.team_a == team:
            goals_for.append(r.ga)
            goals_against.append(r.gb)
        else:
            goals_for.append(r.gb)
            goals_against.append(r.ga)

    return {
        "avg_for": sum(goals_for) / len(goals_for),
        "avg_against": sum(goals_against) / len(goals_against),
        "variance": pd.Series(goals_for).var()
    }

# =========================
# PONDÉRATION RÉCENTE
# =========================
def weighted_matches(df, team):
    games = df[(df.team_a == team) | (df.team_b == team)].tail(10)
    weights = []

    for i in range(len(games)):
        if i >= len(games) - 5:
            weights.append(3)
        elif i >= len(games) - 10:
            weights.append(2)
        else:
            weights.append(1)

    return games, weights

# =========================
# PRÉDICTION PRINCIPALE
# =========================
def predict_match(team_a, team_b):
    df = load_data()
    if len(df) < 5:
        return None

    prof_a = team_profile(team_a, df)
    prof_b = team_profile(team_b, df)

    if not prof_a or not prof_b:
        return None

    exp_a = (prof_a["avg_for"] + prof_b["avg_against"]) / 2
    exp_b = (prof_b["avg_for"] + prof_a["avg_against"]) / 2

    exp_a = round(exp_a)
    exp_b = round(exp_b)

    # Top scores probables
    scores = []
    for i in range(max(0, exp_a-1), exp_a+2):
        for j in range(max(0, exp_b-1), exp_b+2):
            scores.append(f"{i}-{j}")

    counter = Counter(scores)
    total = sum(counter.values())
    top_scores = [(s, round(v/total*100,1)) for s,v in counter.most_common(5)]

    winner = (
        team_a if exp_a > exp_b
        else team_b if exp_b > exp_a
        else "Draw"
    )

    confidence = min(100, int((len(df)/30)*100))

    return {
        "winner": winner,
        "expected_goals": f"{exp_a} - {exp_b}",
        "top_scores": top_scores,
        "confidence": confidence
    }

# =========================
# SCORE DE CONFIANCE
# =========================
def confidence_level(c):
    if c >= 70:
        return "🟢 Élevée"
    if c >= 40:
        return "🟠 Moyenne"
    return "🔴 Faible"
