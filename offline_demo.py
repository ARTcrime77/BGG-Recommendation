#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BGG Offline-Empfehlungssystem Demo

Einfacher Einstiegspunkt für vollständige Offline-Funktionalität
ohne Abhängigkeit von BGG APIs oder Internetverbindung.
"""

import sys
import os

# Füge src-Verzeichnis zum Python-Pfad hinzu
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from offline_recommender import OfflineBGGRecommender, demo_offline_mode
from offline_data import offline_manager


def main():
    """Hauptfunktion für Offline-Demo"""
    print("🔌 BGG Offline-Empfehlungssystem")
    print("=" * 50)
    
    # Prüfe Internetverbindung
    print("🌐 Prüfe Internetverbindung zu BGG...")
    online = offline_manager.check_internet_connectivity(timeout=3)
    
    if online:
        print("✅ BGG ist erreichbar")
        print("💡 Offline-Modus kann trotzdem verwendet werden")
        print("")
        
        choice = input("Möchten Sie den Offline-Modus trotzdem verwenden? (j/n): ").lower().strip()
        if choice not in ['j', 'ja', 'y', 'yes']:
            print("➡️  Verwenden Sie 'python src/main.py' für Online-Modus")
            return
    else:
        print("❌ Keine Internetverbindung zu BGG")
        print("🔌 Starte automatisch im Offline-Modus")
    
    print("")
    
    # Kommandozeilen-Argumente verarbeiten
    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo":
            # Vollständige Demo mit allen Profilen
            demo_offline_mode()
            return
        elif sys.argv[1] == "--list-users":
            # Liste verfügbare Nutzer
            recommender = OfflineBGGRecommender()
            recommender.show_available_profiles()
            return
        elif sys.argv[1].startswith("--user="):
            # Spezifischer Nutzer
            username = sys.argv[1].split("=", 1)[1]
            recommender = OfflineBGGRecommender()
            
            if username in recommender.list_sample_users():
                print(f"🎯 Führe Analyse für {username} durch...")
                recommendations = recommender.run_offline_analysis(username)
                recommender.display_offline_recommendations(recommendations)
            else:
                print(f"❌ Nutzer '{username}' nicht gefunden")
                print("Verfügbare Nutzer:")
                for user in recommender.list_sample_users():
                    print(f"  - {user}")
            return
    
    # Interaktiver Modus
    print("🎮 Interaktiver Offline-Modus")
    print("=" * 30)
    
    recommender = OfflineBGGRecommender()
    
    while True:
        print("\nVerfügbare Optionen:")
        print("1. Verfügbare Nutzerprofile anzeigen")
        print("2. Empfehlungen für spezifischen Nutzer")
        print("3. Demo mit allen Profilen")
        print("4. Beenden")
        
        choice = input("\nWählen Sie eine Option (1-4): ").strip()
        
        if choice == "1":
            recommender.show_available_profiles()
        
        elif choice == "2":
            users = recommender.list_sample_users()
            print(f"\nVerfügbare Nutzer: {', '.join(users)}")
            username = input("Geben Sie einen Nutzernamen ein: ").strip()
            
            if username in users:
                print(f"\n🎯 Führe Analyse für {username} durch...")
                recommendations = recommender.run_offline_analysis(username)
                recommender.display_offline_recommendations(recommendations)
            else:
                print(f"❌ Nutzer '{username}' nicht gefunden")
        
        elif choice == "3":
            print("\n🎯 Führe Demo mit allen Profilen durch...")
            demo_offline_mode()
        
        elif choice == "4":
            print("👋 Auf Wiedersehen!")
            break
        
        else:
            print("❌ Ungültige Auswahl. Bitte wählen Sie 1-4.")


def show_help():
    """Zeigt Hilfe-Informationen"""
    help_text = """
🔌 BGG Offline-Empfehlungssystem - Hilfe

VERWENDUNG:
    python offline_demo.py [OPTIONEN]

OPTIONEN:
    --demo              Führt Demo mit allen verfügbaren Nutzerprofilen durch
    --list-users        Zeigt verfügbare Beispiel-Nutzerprofile an
    --user=USERNAME     Führt Analyse für spezifischen Nutzer durch
    --help              Zeigt diese Hilfe an

BEISPIELE:
    python offline_demo.py
    python offline_demo.py --demo
    python offline_demo.py --list-users
    python offline_demo.py --user=StrategyFan

VERFÜGBARE NUTZERPROFILE:
    - StrategyFan       (Komplexe Strategiespiele)
    - FamilyGamer       (Familienfreundliche Spiele)
    - ThematicPlayer    (Atmosphärische, thematische Spiele)

OFFLINE-FUNKTIONEN:
    ✅ 1000+ Spiele in Offline-Datenbank
    ✅ 3 Beispiel-Nutzerprofile mit verschiedenen Präferenzen
    ✅ Vollständiges ML-Training und Empfehlungen
    ✅ Feature-Caching für bessere Performance
    ✅ Keine Internetverbindung erforderlich

SYSTEM-ANFORDERUNGEN:
    - Python 3.7+
    - Installierte Abhängigkeiten (siehe requirements.txt)
    - Verfügbarer Speicherplatz für Cache (~50MB)
"""
    print(help_text)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
        show_help()
    else:
        main()