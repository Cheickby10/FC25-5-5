import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import json
import os

class IAEngine:
    """
    IAEngine complet pour FC25 5x5 Rush
    - Entraînement automatique sur historique CSV
    - Mémoire persistante par équipe
    - Pondération des matchs récents
    - Calcul du score exact, top 5 scores probables
    - Fiabilité réelle basée sur variance et cohérence des équipes
    - Détection de matchs répétitifs/incestueux
    - Historique complet pour apprentissage continu
    """

    def __init__(self, csv_path, memory_path="data/memory.json"):
        self.csv_path = csv_path
        self.memory_path = memory_path
        self.df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame(columns=["team_a","team_b","ga","gb","date"])
        self.team_stats = {}  # Mémoire par équipe
        self.score_counts = None
        self.recent_weight = 1.5  # poids des matchs récents
        self.top_n_scores = 5
        self.load_memory()
        self.prepare_stats()

    # -----------------------------
    # Mémoire persistante
    # -----------------------------
    def load_memory(self):
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r") as f:
                self.memory = json.load(f)
        else:
            self.memory = {}

    def save_memory(self):
        with open(self.memory_path, "w") as f:
            json.dump(self.memory, f, indent=2)

    # -----------------------------
    # Préparer les statistiques par équipe
    # -----------------------------
    def prepare_stats(self):
        df = self.df.copy()
        if 'date' not in df.columns:
            df['date'] = datetime.today().strftime("%Y-%m-%d")
        df['weight'] = 1.0

        # Pondération des matchs récents
        dates = pd.to_datetime(df['date'])
        days_ago = (datetime.today() - dates).dt.days
        df['weight'] = np.exp(-days_ago/30) * self.recent_weight  # plus récent = plus influent

        # Mémoire des équipes
        teams = set(df['team_a']).union(set(df['team_b']))
        self.team_stats = {}
        for team in teams:
            a_stats = df[df['team_a']==team]
            b_stats = df[df['team_b']==team]
            poids_total = a_stats['weight'].sum() + b_stats['weight'].sum()
            buts_marques = (a_stats['ga']*a_stats['weight']).sum() + (b_stats['gb']*b_stats['weight']).sum()
            buts_encaisses = (a_stats['gb']*a_stats['weight']).sum() + (b_stats['ga']*b_stats['weight']).sum()
            self.team_stats[team] = {
                "buts_marques_moy": buts_marques/poids_total if poids_total>0 else 0,
                "buts_encaisses_moy": buts_encaisses/poids_total if poids_total>0 else 0,
                "matchs_joues": len(a_stats)+len(b_stats)
            }
            # Sauvegarde dans mémoire persistante
            self.memory[team] = self.team_stats[team]
        self.save_memory()

        # Historique des scores
        self.score_counts = Counter([(row['ga'],row['gb']) for idx,row in df.iterrows()])

    # -----------------------------
    # Détection de matchs incestueux
    # -----------------------------
    def detect_incestuous(self, team_a, team_b):
        df_recent = self.df.tail(20)  # derniers 20 matchs
        repet = df_recent[((df_recent['team_a']==team_a)&(df_recent['team_b']==team_b))|
                          ((df_recent['team_a']==team_b)&(df_recent['team_b']==team_a))]
        return len(repet)>2  # trop répété

    # -----------------------------
    # Entraîner IA (réentraînement automatique)
    # -----------------------------
    def train(self):
        self.prepare_stats()
        # Score counts déjà calculé
        # Possibilité d'ajouter apprentissage ML ou xG ici plus tard

    # -----------------------------
    # Prédire un match
    # -----------------------------
    def predict(self, team_a, team_b):
        # Détection matchs incestueux
        if self.detect_incestuous(team_a, team_b):
            incest_warning = True
        else:
            incest_warning = False

        # Score attendu par moyenne des stats
        stats_a = self.team_stats.get(team_a, {"buts_marques_moy":1,"buts_encaisses_moy":1})
        stats_b = self.team_stats.get(team_b, {"buts_marques_moy":1,"buts_encaisses_moy":1})
        score_a = round((stats_a['buts_marques_moy'] + stats_b['buts_encaisses_moy'])/2)
        score_b = round((stats_b['buts_marques_moy'] + stats_a['buts_encaisses_moy'])/2)

        # Top N scores probables
        top_scores = self.score_counts.most_common(self.top_n_scores)
        total = sum(self.score_counts.values())
        if total>0:
            top5_prob = [ (s, c/total) for s,c in top_scores ]
        else:
            # fallback autour du score attendu
            top5_prob = [
                ((score_a, score_b),0.4),
                ((score_a+1, score_b),0.2),
                ((score_a, score_b+1),0.2),
                ((max(score_a-1,0), score_b),0.1),
                ((score_a, max(score_b-1,0)),0.1)
            ]

        # Fiabilité : inverse de variance ajustée
        if self.df.empty:
            confidence = 0.5
        else:
            variance = self.df[["ga","gb"]].stack().var()
            confidence = max(0.1, min(0.99, 1/(1+variance)))

        return (score_a, score_b), confidence, top5_prob, incest_warning

    # -----------------------------
    # Ajouter un match manuel à l'historique
    # -----------------------------
    def add_match(self, team_a, team_b, score_a, score_b, date=None):
        if date is None:
            date = datetime.today().strftime("%Y-%m-%d")
        new_row = {"team_a":team_a, "team_b":team_b, "ga":score_a, "gb":score_b, "date":date}
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        # Sauvegarde CSV
        self.df.to_csv(self.csv_path, index=False)
        # Réentraîner IA
        self.train()
