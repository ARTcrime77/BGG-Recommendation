# -*- coding: utf-8 -*-
"""
HTML Report Generator for BGG ML-Empfehlungssystem
"""

import os
from datetime import datetime

class ReportGenerator:
    """
    Erstellt einen HTML-Report aus den generierten Plots und Daten.
    """

    @staticmethod
    def create_html_report(recommendations, user_preferences, plot_paths, save_dir=""):
        """
        Erstellt einen HTML-Report.

        Args:
            recommendations (list): Die Liste der Empfehlungs-Dictionaries.
            user_preferences (dict): Das Dictionary mit den Nutzerpräferenzen.
            plot_paths (dict): Ein Dictionary, das Plot-Typen auf ihre Dateipfade abbildet.
            save_dir (str): Das Verzeichnis, in dem der Report gespeichert werden soll.
        """
        if not plot_paths:
            print("⚠️ Keine Plot-Pfade für den Report gefunden.")
            return

        report_path = os.path.join(save_dir, "bgg_recommendation_report.html")
        
        # Sicherstellen, dass die Pfade relativ zum Report-Verzeichnis sind
        relative_plot_paths = {key: os.path.relpath(path, save_dir) for key, path in plot_paths.items()}

        # HTML-Struktur
        html_content = f"""
        <!DOCTYPE html>
        <html lang="de">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>BGG Empfehlungs-Report</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; background-color: #f8f9fa; color: #212529; }}
                .container {{ max-width: 1200px; margin: 20px auto; padding: 20px; background-color: #ffffff; box-shadow: 0 0 15px rgba(0,0,0,0.1); border-radius: 8px; }}
                header {{ text-align: center; border-bottom: 1px solid #dee2e6; padding-bottom: 20px; margin-bottom: 30px; }}
                header h1 {{ margin: 0; font-size: 2.5em; color: #343a40; }}
                header p {{ margin: 5px 0 0; color: #6c757d; }}
                .plot-section {{ margin-bottom: 40px; padding: 20px; border: 1px solid #e9ecef; border-radius: 8px; background-color: #fff; }}
                .plot-section h2 {{ font-size: 1.8em; color: #495057; border-bottom: 1px solid #dee2e6; padding-bottom: 10px; margin-top: 0; }}
                .plot-section p.description {{ color: #495057; line-height: 1.6; font-size: 1.1em; }}
                .plot-section img {{ max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 4px 8px rgba(0,0,0,0.05); }}
                footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #dee2e6; color: #6c757d; }}
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>BGG Empfehlungs-Report</h1>
                    <p>Erstellt am: {datetime.now().strftime('%d.%m.%Y um %H:%M:%S')}</p>
                </header>

                {ReportGenerator._generate_plot_section(
                    "Sammlungs-Statistik",
                    relative_plot_paths.get('collection_stats'),
                    "Diese Visualisierung bietet einen Überblick über Ihre aktuelle Sammlung und Spielaktivitäten. Sie zeigt die Verteilung Ihrer bewerteten vs. besessenen Spiele, wie Sie Spiele bewerten, Ihre meistgespielten Spiele und Ihre Spielaktivität über die Zeit (sofern Daten verfügbar sind)."
                )}

                {ReportGenerator._generate_plot_section(
                    "Nutzerprofil-Analyse",
                    relative_plot_paths.get('user_prefs'),
                    "Diese Visualisierung zeigt die Präferenzen, die das Modell über Sie gelernt hat. Das <strong>Radar-Diagramm</strong> links zeigt Ihre Affinität zu den Top-Spielkategorien und -mechaniken. Je weiter ein Punkt vom Zentrum entfernt ist, desto stärker ist die Präferenz. Das <strong>Balkendiagramm</strong> rechts zeigt Ihre idealen numerischen Attribute für ein Spiel, wie Komplexität und Spieldauer."
                )}

                {ReportGenerator._generate_plot_section(
                    "Top-Empfehlungen nach Ähnlichkeit",
                    relative_plot_paths.get('similarity'),
                    "Dieses Diagramm hebt Ihre Top-10-Empfehlungen hervor. Das <strong>Balkendiagramm</strong> links ordnet sie nach ihrem Ähnlichkeits-Score – ein Maß dafür, wie gut sie mit Ihrem Profil übereinstimmen. Der <strong>Streu-Plot</strong> rechts zeigt diese Empfehlungen nach ihrer Ähnlichkeit im Vergleich zu ihrer Gesamtbewertung auf BGG und hilft Ihnen, Spiele zu finden, die sowohl gut passen als auch hoch bewertet sind."
                )}

                {ReportGenerator._generate_plot_section(
                    "Heatmap: Ära und Komplexität",
                    relative_plot_paths.get('era_complexity'),
                    "Diese Heatmap zeigt, wo Ihre empfohlenen Spiele in Bezug auf ihr Erscheinungsdatum und ihre Komplexität einzuordnen sind. Die Zahl und Farbintensität jeder Zelle geben an, wie viele Ihrer Empfehlungen in diese spezifische Ära und Komplexitätsstufe fallen. Dies kann Ihnen helfen zu sehen, ob Sie zu modernen Experten-Spielen oder zu älteren, leichteren Titeln tendieren."
                )}

                {ReportGenerator._generate_plot_section(
                    "Treemap: Autoren und Verlage",
                    relative_plot_paths.get('creators'),
                    "Diese Treemap visualisiert die häufigsten Spieleautoren (links) und Verlage (rechts) innerhalb Ihrer Empfehlungen. Die Größe jedes Rechtecks ist proportional zur Anzahl der empfohlenen Spiele von diesem Schöpfer. Dies ist eine großartige Möglichkeit, neue Lieblingsautoren oder Verlage zu entdecken, deren Arbeit durchweg Ihrem Geschmack entspricht."
                )}

                {ReportGenerator._generate_plot_section(
                    "Verteilung der Bewertungen",
                    relative_plot_paths.get('ratings'),
                    "Diese Diagramme analysieren die Verteilung der BGG-Bewertungen für alle Spiele in der Datenbank. Sie bieten Kontext für Ihre Empfehlungen, indem sie zeigen, wo diese im breiteren Spektrum der Brettspielbewertungen liegen. Wenn Ihre Spielesammlung analysiert wurde, wird sie mit der globalen Verteilung verglichen."
                )}

                {ReportGenerator._generate_plot_section(
                    "Technische Feature-Analyse",
                    relative_plot_paths.get('features'),
                    "Dieses technische Dashboard bietet einen tieferen Einblick in die vom Machine-Learning-Modell verwendeten Daten. Es enthält eine <strong>Korrelationsmatrix</strong> für numerische Merkmale, die <strong>häufigsten Merkmale</strong> (wie spezifische Mechaniken) und die Merkmale mit der <strong>höchsten Varianz</strong>, die den größten Einfluss auf das Modell haben."
                )}

                <footer>
                    <p>Report generiert vom BGG ML-Empfehlungssystem.</p>
                </footer>
            </div>
        </body>
        </html>
        """

        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"✅ HTML-Report erfolgreich gespeichert: {report_path}")
        except IOError as e:
            print(f"❌ Fehler beim Speichern des HTML-Reports: {e}")

    @staticmethod
    def _generate_plot_section(title, img_path, description):
        """Hilfsfunktion zum Generieren eines Abschnitts im Report."""
        if not img_path or not os.path.exists(img_path):
            return ""
        
        return f"""
        <section class="plot-section">
            <h2>{title}</h2>
            <p class="description">{description}</p>
            <img src="{img_path}" alt="{title}">
        </section>
        """

