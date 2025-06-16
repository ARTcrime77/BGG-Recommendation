"""
BGG Recommendation Visualizer

Visualisiert Empfehlungsergebnisse, Bewertungsverteilungen und Feature-Analysen
für das BGG ML-Empfehlungssystem.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Deutsche Schriftarten und Stil
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

class BGGVisualizer:
    def __init__(self):
        """Initialisiert den BGG Visualizer"""
        self.style_setup()
    
    def style_setup(self):
        """Konfiguriert den visuellen Stil"""
        plt.style.use('default')
        sns.set_palette("husl")
        
    def plot_recommendation_similarity(self, recommendations_df, user_games=None, save_path=None, show_gui=True):
        """
        Erstellt ein Ähnlichkeitsdiagramm der Empfehlungen
        
        Args:
            recommendations_df: DataFrame mit Empfehlungen und Similarity-Scores
            user_games: Liste der Nutzerspiele (optional)
            save_path: Pfad zum Speichern der Grafik (optional)
        """
        if recommendations_df.empty:
            print("⚠️ Keine Empfehlungen zum Visualisieren vorhanden.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Horizontales Balkendiagramm der Ähnlichkeits-Scores
        game_names = recommendations_df['name'].head(10)  # Top 10
        similarity_scores = recommendations_df['similarity_score'].head(10)
        
        # Farben basierend auf Ähnlichkeit
        colors = plt.cm.RdYlGn(similarity_scores / similarity_scores.max())
        
        bars = ax1.barh(range(len(game_names)), similarity_scores, color=colors)
        ax1.set_yticks(range(len(game_names)))
        ax1.set_yticklabels([name[:30] + '...' if len(name) > 30 else name 
                            for name in game_names], fontsize=9)
        ax1.set_xlabel('Ähnlichkeits-Score')
        ax1.set_title('Top 10 Empfehlungen nach Ähnlichkeit', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Werte auf Balken anzeigen
        for i, (bar, score) in enumerate(zip(bars, similarity_scores)):
            ax1.text(score + 0.01, i, f'{score:.3f}', 
                    va='center', ha='left', fontsize=8)
        
        # 2. Scatter Plot: Bewertung vs. Ähnlichkeit
        if 'rating' in recommendations_df.columns:
            ratings = recommendations_df['rating'].head(15)
            similarities = recommendations_df['similarity_score'].head(15)
            
            scatter = ax2.scatter(similarities, ratings, 
                                c=range(len(similarities)), 
                                cmap='viridis', s=100, alpha=0.7, 
                                edgecolors='black', linewidth=1)
            
            # Beste Empfehlungen hervorheben (hohe Ähnlichkeit + hohe Bewertung)
            for i, (sim, rat, name) in enumerate(zip(similarities, ratings, 
                                                   recommendations_df['name'].head(15))):
                if sim > similarities.mean() and rat > ratings.mean():
                    ax2.annotate(name[:20], (sim, rat), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=7, alpha=0.8)
            
            ax2.set_xlabel('Ähnlichkeits-Score')
            ax2.set_ylabel('BGG Bewertung')
            ax2.set_title('Empfehlungen: Ähnlichkeit vs. Bewertung', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Colorbar für Ranking
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Empfehlungs-Ranking')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Ähnlichkeitsdiagramm gespeichert: {save_path}")
        
        if show_gui:
            try:
                plt.show(block=False)  # Non-blocking show
                plt.pause(0.1)  # Kurze Pause für Rendering
            except:
                print("⚠️ GUI-Anzeige nicht verfügbar - Plot nur als Datei gespeichert")
        
        plt.close()  # Schließe Figure um Speicher freizugeben
    
    def plot_rating_distribution(self, games_data, user_collection=None, save_path=None, show_gui=True):
        """
        Visualisiert die Bewertungsverteilung der Spiele
        
        Args:
            games_data: Liste oder DataFrame mit Spieledaten (muss 'rating' enthalten)
            user_collection: Nutzerspiele für Vergleich (optional)
            save_path: Pfad zum Speichern der Grafik (optional)
        """
        if isinstance(games_data, list):
            ratings = [game.get('rating', 0) for game in games_data 
                      if game.get('rating') is not None and game.get('rating', 0) > 0]
        else:
            ratings = games_data['rating'].dropna()
            ratings = [r for r in ratings if r is not None and r > 0]
        
        if len(ratings) == 0:
            print("⚠️ Keine Bewertungsdaten zum Visualisieren vorhanden.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Histogramm der Bewertungsverteilung
        ax1.hist(ratings, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(ratings), color='red', linestyle='--', 
                   label=f'Durchschnitt: {np.mean(ratings):.2f}')
        ax1.axvline(np.median(ratings), color='orange', linestyle='--', 
                   label=f'Median: {np.median(ratings):.2f}')
        ax1.set_xlabel('BGG Bewertung')
        ax1.set_ylabel('Anzahl Spiele')
        ax1.set_title('Verteilung der BGG Bewertungen', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box Plot
        ax2.boxplot(ratings, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax2.set_ylabel('BGG Bewertung')
        ax2.set_title('Bewertungs-Boxplot', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Bewertungskategorien
        rating_categories = pd.cut(ratings, 
                                 bins=[0, 5.5, 6.5, 7.5, 8.5, 10], 
                                 labels=['Schlecht\n(≤5.5)', 'Okay\n(5.5-6.5)', 
                                        'Gut\n(6.5-7.5)', 'Sehr gut\n(7.5-8.5)', 
                                        'Exzellent\n(>8.5)'])
        
        category_counts = rating_categories.value_counts()
        colors = ['#ff9999', '#ffcc99', '#99ccff', '#99ff99', '#99ff99']
        
        wedges, texts, autotexts = ax3.pie(category_counts.values, 
                                          labels=category_counts.index,
                                          colors=colors, autopct='%1.1f%%',
                                          startangle=90)
        ax3.set_title('Bewertungskategorien', fontweight='bold')
        
        # 4. Vergleich mit Nutzersammlung (falls vorhanden)
        if user_collection:
            if isinstance(user_collection, list):
                user_ratings = [game.get('rating', 0) for game in user_collection 
                              if game.get('rating') is not None and game.get('rating', 0) > 0]
            else:
                user_ratings = user_collection['rating'].dropna()
                user_ratings = [r for r in user_ratings if r is not None and r > 0]
            
            if len(user_ratings) > 0:
                ax4.hist([ratings, user_ratings], bins=20, alpha=0.7, 
                        label=['Alle Spiele', 'Nutzersammlung'],
                        color=['lightblue', 'lightcoral'])
                ax4.set_xlabel('BGG Bewertung')
                ax4.set_ylabel('Anzahl Spiele')
                ax4.set_title('Bewertungsvergleich: Alle vs. Nutzer', fontweight='bold')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'Keine Nutzerbewertungen\nverfügbar', 
                        ha='center', va='center', transform=ax4.transAxes,
                        fontsize=12, bbox=dict(boxstyle="round,pad=0.3", 
                                             facecolor="lightgray"))
                ax4.set_title('Nutzersammlung', fontweight='bold')
        else:
            # Statistiken anzeigen
            stats_text = f"""Bewertungsstatistiken:
            
Anzahl Spiele: {len(ratings)}
Durchschnitt: {np.mean(ratings):.2f}
Median: {np.median(ratings):.2f}
Standardabw.: {np.std(ratings):.2f}
Min: {np.min(ratings):.2f}
Max: {np.max(ratings):.2f}

Quartile:
25%: {np.percentile(ratings, 25):.2f}
75%: {np.percentile(ratings, 75):.2f}"""
            
            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            ax4.set_title('Statistiken', fontweight='bold')
            ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Bewertungsverteilung gespeichert: {save_path}")
        
        if show_gui:
            try:
                plt.show(block=False)
                plt.pause(0.1)
            except:
                print("⚠️ GUI-Anzeige nicht verfügbar - Plot nur als Datei gespeichert")
        
        plt.close()
    
    def plot_feature_analysis(self, feature_matrix, feature_names, games_data=None, save_path=None, show_gui=True):
        """
        Erstellt Feature-Analyse-Plots
        
        Args:
            feature_matrix: NumPy Array oder DataFrame mit Features
            feature_names: Liste der Feature-Namen
            games_data: Spieledaten für zusätzliche Analysen (optional)
            save_path: Pfad zum Speichern der Grafik (optional)
        """
        if isinstance(feature_matrix, pd.DataFrame):
            df = feature_matrix
        else:
            df = pd.DataFrame(feature_matrix, columns=feature_names)
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Feature-Korrelationsmatrix (nur numerische Features)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            ax1 = plt.subplot(2, 3, 1)
            corr_matrix = df[numeric_cols].corr()
            
            # Nur die wichtigsten Korrelationen anzeigen
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', 
                       center=0, square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
            ax1.set_title('Feature-Korrelationen', fontweight='bold')
        
        # 2. Top Feature-Häufigkeiten (für kategorische Features)
        ax2 = plt.subplot(2, 3, 2)
        
        # Identifiziere One-Hot-Encoded Features
        categorical_features = [col for col in df.columns if df[col].dtype == 'uint8' or 
                              (df[col].nunique() == 2 and set(df[col].unique()).issubset({0, 1}))]
        
        if categorical_features:
            # Top 15 aktivste Features
            feature_sums = df[categorical_features].sum().sort_values(ascending=False).head(15)
            
            bars = ax2.barh(range(len(feature_sums)), feature_sums.values)
            ax2.set_yticks(range(len(feature_sums)))
            ax2.set_yticklabels([name.replace('_', ' ').title()[:25] + '...' 
                               if len(name) > 25 else name.replace('_', ' ').title() 
                               for name in feature_sums.index], fontsize=8)
            ax2.set_xlabel('Anzahl Spiele')
            ax2.set_title('Top 15 Feature-Häufigkeiten', fontweight='bold')
            
            # Werte anzeigen
            for i, v in enumerate(feature_sums.values):
                ax2.text(v + 0.5, i, str(int(v)), va='center', fontsize=8)
        
        # 3. Numerische Feature-Verteilungen
        ax3 = plt.subplot(2, 3, 3)
        
        # Wichtige numerische Features
        key_numeric = [col for col in numeric_cols if any(keyword in col.lower() 
                      for keyword in ['rating', 'complexity', 'players', 'time', 'year'])]
        
        if key_numeric:
            # Normalisierte Box Plots
            normalized_data = []
            labels = []
            
            for col in key_numeric[:5]:  # Top 5
                if df[col].std() > 0:  # Nur Features mit Varianz
                    normalized_values = (df[col] - df[col].mean()) / df[col].std()
                    normalized_data.append(normalized_values)
                    labels.append(col.replace('_', ' ').title())
            
            if normalized_data:
                ax3.boxplot(normalized_data, labels=labels)
                ax3.set_ylabel('Normalisierte Werte (Z-Score)')
                ax3.set_title('Numerische Feature-Verteilungen', fontweight='bold')
                ax3.tick_params(axis='x', rotation=45)
                ax3.grid(True, alpha=0.3)
        
        # 4. Feature-Wichtigkeit (basierend auf Varianz)
        ax4 = plt.subplot(2, 3, 4)
        
        feature_variance = df.var().sort_values(ascending=False).head(15)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(feature_variance)))
        bars = ax4.bar(range(len(feature_variance)), feature_variance.values, color=colors)
        ax4.set_xticks(range(len(feature_variance)))
        ax4.set_xticklabels([name.replace('_', ' ')[:10] + '...' 
                           if len(name) > 10 else name.replace('_', ' ') 
                           for name in feature_variance.index], rotation=45, ha='right')
        ax4.set_ylabel('Varianz')
        ax4.set_title('Top 15 Features nach Varianz', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Spiele-Komplexität vs. Bewertung (falls verfügbar)
        ax5 = plt.subplot(2, 3, 5)
        
        complexity_cols = [col for col in df.columns if 'complexity' in col.lower()]
        rating_cols = [col for col in df.columns if 'rating' in col.lower()]
        
        if complexity_cols and rating_cols and games_data:
            try:
                if isinstance(games_data, list):
                    complexities = [game.get('complexity', 0) for game in games_data 
                                  if game.get('complexity') is not None and game.get('complexity', 0) > 0]
                    ratings = [game.get('rating', 0) for game in games_data 
                             if game.get('rating') is not None and game.get('rating', 0) > 0]
                else:
                    complexities = games_data['complexity'].dropna() if 'complexity' in games_data.columns else []
                    complexities = [c for c in complexities if c is not None and c > 0]
                    ratings = games_data['rating'].dropna() if 'rating' in games_data.columns else []
                    ratings = [r for r in ratings if r is not None and r > 0]
                
                if len(complexities) > 0 and len(ratings) > 0 and len(complexities) == len(ratings):
                    scatter = ax5.scatter(complexities, ratings, alpha=0.6, s=50)
                    
                    # Trendlinie
                    z = np.polyfit(complexities, ratings, 1)
                    p = np.poly1d(z)
                    ax5.plot(complexities, p(complexities), "r--", alpha=0.8)
                    
                    ax5.set_xlabel('Komplexität')
                    ax5.set_ylabel('BGG Bewertung')
                    ax5.set_title('Komplexität vs. Bewertung', fontweight='bold')
                    ax5.grid(True, alpha=0.3)
                else:
                    ax5.text(0.5, 0.5, 'Komplexitätsdaten\nnicht verfügbar', 
                           ha='center', va='center', transform=ax5.transAxes)
            except Exception as e:
                ax5.text(0.5, 0.5, f'Fehler beim Laden\nder Komplexitätsdaten', 
                        ha='center', va='center', transform=ax5.transAxes)
        else:
            ax5.text(0.5, 0.5, 'Komplexitätsdaten\nnicht verfügbar', 
                    ha='center', va='center', transform=ax5.transAxes,
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        ax5.set_title('Komplexitäts-Analyse', fontweight='bold')
        
        # 6. Feature-Zusammenfassung
        ax6 = plt.subplot(2, 3, 6)
        
        summary_text = f"""Feature-Matrix Zusammenfassung:

Dimensionen: {df.shape[0]} × {df.shape[1]}

Feature-Typen:
• Numerisch: {len(numeric_cols)}
• Kategorisch: {len(categorical_features)}

Speicherverbrauch:
{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB

Null-Werte: {df.isnull().sum().sum()}

Top Feature-Kategorien:
{len([col for col in df.columns if 'category' in col.lower()])} Kategorien
{len([col for col in df.columns if 'mechanic' in col.lower()])} Mechaniken
{len([col for col in df.columns if 'designer' in col.lower()])} Designer"""
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        ax6.set_title('Feature-Matrix Info', fontweight='bold')
        ax6.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Feature-Analyse gespeichert: {save_path}")
        
        if show_gui:
            try:
                plt.show(block=False)
                plt.pause(0.1)
            except:
                print("⚠️ GUI-Anzeige nicht verfügbar - Plot nur als Datei gespeichert")
        
        plt.close()
    
    def create_dashboard(self, recommendations_df, games_data, feature_matrix, 
                        feature_names, user_collection=None, save_path=None):
        """
        Erstellt ein komplettes Dashboard mit allen Visualisierungen
        
        Args:
            recommendations_df: DataFrame mit Empfehlungen
            games_data: Spieledaten
            feature_matrix: Feature-Matrix
            feature_names: Feature-Namen
            user_collection: Nutzersammlung (optional)
            save_path: Pfad zum Speichern (optional)
        """
        print("\n📊 Erstelle BGG Empfehlungs-Dashboard...")
        
        # Einzelne Plots erstellen
        print("1️⃣ Ähnlichkeitsdiagramm...")
        self.plot_recommendation_similarity(recommendations_df, user_collection)
        
        print("2️⃣ Bewertungsverteilung...")
        self.plot_rating_distribution(games_data, user_collection)
        
        print("3️⃣ Feature-Analyse...")
        self.plot_feature_analysis(feature_matrix, feature_names, games_data)
        
        print("✅ Dashboard komplett erstellt!")


def demo_visualizer():
    """Demo-Funktion für den BGG Visualizer"""
    print("🎯 BGG Visualizer Demo")
    
    # Beispieldaten erstellen
    np.random.seed(42)
    
    # Mock Recommendations
    recommendations = pd.DataFrame({
        'name': [f'Spiel {i+1}' for i in range(20)],
        'similarity_score': np.random.beta(2, 5, 20),
        'rating': np.random.normal(7.2, 1.2, 20)
    })
    
    # Mock Games Data
    games_data = pd.DataFrame({
        'rating': np.random.normal(6.8, 1.5, 500),
        'complexity': np.random.uniform(1, 5, 500)
    })
    
    # Mock Feature Matrix
    feature_matrix = np.random.rand(100, 50)
    feature_names = [f'feature_{i}' for i in range(50)]
    
    # Visualizer testen
    viz = BGGVisualizer()
    viz.create_dashboard(recommendations, games_data, feature_matrix, feature_names)


if __name__ == "__main__":
    demo_visualizer()