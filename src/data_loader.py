"""
Datenlade-Funktionen für das BGG ML-Empfehlungssystem
Verwendet CacheManager für einheitliche Cache-Verwaltung
"""

import requests
import xml.etree.ElementTree as ET
import json
import os
import time
import re
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from collections import Counter

from config import (
    BGG_API_BASE_URL,
    BGG_BROWSE_URL,
    API_DELAY,
    BATCH_SIZE,
    USER_AGENT,
    SCRAPING_DELAY,
    TARGET_TOP_GAMES,
    MAX_SCRAPING_PAGES,
    SHOW_PROGRESS_EVERY
)
from cache_manager import cache_manager


class BGGDataLoader:
    def __init__(self):
        """Initialisiert BGG Data Loader mit CacheManager"""
        self.cache = cache_manager
    
    def should_update_cache(self, filepath, max_age_days=None):
        """Delegiert an CacheManager (Kompatibilität)"""
        return self.cache.should_update_cache(filepath, max_age_days)
    
    def ask_user_update_choice(self, cache_type):
        """Delegiert an CacheManager (Kompatibilität)"""
        return self.cache.ask_user_update_choice(cache_type)
    
    def remove_duplicates_from_games(self, games_list):
        """Entfernt Duplikate basierend auf Spiel-ID"""
        seen_ids = set()
        unique_games = []
        duplicates_found = 0
        
        for game in games_list:
            game_id = game['id']
            if game_id not in seen_ids:
                seen_ids.add(game_id)
                unique_games.append(game)
            else:
                duplicates_found += 1
                print(f"   🔍 Duplikat entfernt: {game['name']} (ID: {game_id})")
        
        if duplicates_found > 0:
            msg = f"✓ {duplicates_found} Duplikate entfernt. "
            msg += f"{len(unique_games)} eindeutige Spiele übrig."
            print(msg)
        else:
            print(f"✓ Keine Duplikate gefunden. {len(unique_games)} eindeutige Spiele.")
        
        return unique_games
    
    def scrape_bgg_top_games(self):
        """Scrapt BGG Top-Spiele bis TARGET_TOP_GAMES eindeutige Spiele erreicht sind"""
        print(f"🕷️  Lade BGG Top-Spiele (Ziel: {TARGET_TOP_GAMES} eindeutige)...")
        
        all_games = []
        unique_games = []
        seen_ids = set()
        
        try:
            for page in range(1, MAX_SCRAPING_PAGES + 1):
                print(f"  Lade Seite {page}/{MAX_SCRAPING_PAGES}...")
                
                params = {
                    'sort': 'rank',
                    'page': page
                }
                
                headers = {
                    'User-Agent': USER_AGENT
                }
                
                response = requests.get(BGG_BROWSE_URL+str(page), params=params, headers=headers)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Finde Spiel-Links (BGG-spezifisches HTML parsing)
                    game_rows = soup.find_all('tr', {'id': re.compile(r'row_')})
                    
                    page_games = []
                    for row in game_rows:
                        try:
                            # Rang extrahieren
                            rank_cell = row.find('td', class_='collection_rank')
                            if rank_cell:
                                rank_text = rank_cell.get_text().strip()
                                rank_match = re.search(r'\d+', rank_text)
                                if rank_match:
                                    rank = int(rank_match.group())
                                else:
                                    continue
                            else:
                                continue
                            
                            # Spiel-ID und Name extrahieren
                            name_cell = row.find('td', class_='collection_thumbnail')
                            if name_cell:
                                link = name_cell.find('a')
                                if link and 'href' in link.attrs:
                                    href = link['href']
                                    game_id_match = re.search(r'/boardgame/(\d+)/', href)
                                    if game_id_match:
                                        game_id = int(game_id_match.group(1))
                                        
                                        # Name extrahieren
                                        img = link.find('img')
                                        name = img.get('alt', 'Unknown') if img else 'Unknown'
                                        
                                        page_games.append({
                                            'rank': rank,
                                            'id': game_id,
                                            'name': name
                                        })
                        except Exception as e:
                            # Einzelne Parsing-Fehler ignorieren
                            continue
                    
                    all_games.extend(page_games)
                    
                    # Duplikate sofort entfernen und eindeutige Spiele sammeln
                    for game in page_games:
                        if game['id'] not in seen_ids:
                            seen_ids.add(game['id'])
                            unique_games.append(game)
                    
                    print(f"    Seite {page}: {len(page_games)} Spiele gefunden, {len(unique_games)} eindeutige bisher")
                    
                    # Prüfen ob Ziel erreicht
                    if len(unique_games) >= TARGET_TOP_GAMES:
                        print(f"✓ Ziel von {TARGET_TOP_GAMES} eindeutigen Spielen erreicht!")
                        break
                
                time.sleep(SCRAPING_DELAY)
            
            if unique_games:
                # Nach Rang sortieren
                unique_games.sort(key=lambda x: x['rank'])
                
                # Auf Zielanzahl begrenzen
                final_games = unique_games[:TARGET_TOP_GAMES]
                
                # Speichern
                self.save_top_games_cache(final_games, len(all_games), len(unique_games))
                
                print(f"✓ {len(final_games)} eindeutige Top-Spiele erfolgreich geladen und gespeichert")
                if len(all_games) > len(unique_games):
                    print(f"  (Von {len(all_games)} gescrapten Spielen waren {len(all_games) - len(unique_games)} Duplikate)")
                return final_games
            else:
                print("⚠️  Keine Spiele über Web-Scraping gefunden. Verwende Fallback...")
                return self.get_fallback_top_games()
                
        except Exception as e:
            print(f"❌ Fehler beim Scraping: {e}")
            print("🔄 Verwende Fallback Top-Spiele...")
            return self.get_fallback_top_games()
    
    def get_fallback_top_games(self):
        """Fallback-Liste mit bekannten Top-Spielen - erweitert auf TARGET_TOP_GAMES"""
        print(f"🎲 Erstelle Fallback-Liste mit {TARGET_TOP_GAMES} Spielen...")
        
        # Basis-Liste mit bekannten Top-Spielen
        base_games = [
            {'rank': 1, 'id': 174430, 'name': 'Gloomhaven'},
            {'rank': 2, 'id': 233078, 'name': 'Twilight Imperium: Fourth Edition'},
            {'rank': 3, 'id': 167791, 'name': 'Terraforming Mars'},
            {'rank': 4, 'id': 220308, 'name': 'Gaia Project'},
            {'rank': 5, 'id': 173346, 'name': '7 Wonders Duel'},
            {'rank': 6, 'id': 169786, 'name': 'Scythe'},
            {'rank': 7, 'id': 161936, 'name': 'Pandemic Legacy: Season 1'},
            {'rank': 8, 'id': 182028, 'name': 'Through the Ages: A New Story of Civilization'},
            {'rank': 9, 'id': 12333, 'name': 'Twilight Struggle'},
            {'rank': 10, 'id': 68448, 'name': '7 Wonders'},
            {'rank': 11, 'id': 36218, 'name': 'Dominion'},
            {'rank': 12, 'id': 31260, 'name': 'Agricola'},
            {'rank': 13, 'id': 13, 'name': 'Catan'},
            {'rank': 14, 'id': 68, 'name': 'Acquire'},
            {'rank': 15, 'id': 822, 'name': 'Carcassonne'},
            {'rank': 16, 'id': 21, 'name': 'Ticket to Ride'},
            {'rank': 17, 'id': 39856, 'name': 'Dixit'},
            {'rank': 18, 'id': 30549, 'name': 'Pandemic'},
            {'rank': 19, 'id': 84876, 'name': 'Splendor'},
            {'rank': 20, 'id': 148949, 'name': 'Azul'},
            {'rank': 21, 'id': 230802, 'name': 'Azul: Stained Glass of Sintra'},
            {'rank': 22, 'id': 256916, 'name': 'Brass: Birmingham'},
            {'rank': 23, 'id': 28720, 'name': 'Brass'},
            {'rank': 24, 'id': 266192, 'name': 'Wingspan'},
            {'rank': 25, 'id': 245654, 'name': 'Spirit Island'},
            {'rank': 26, 'id': 205059, 'name': 'Ark Nova'},
            {'rank': 27, 'id': 286096, 'name': 'Brass: Lancashire'},
            {'rank': 28, 'id': 120677, 'name': 'Terra Mystica'},
            {'rank': 29, 'id': 102794, 'name': 'Caverna: The Cave Farmers'},
            {'rank': 30, 'id': 62219, 'name': 'Hanabi'},
            {'rank': 31, 'id': 146508, 'name': 'Codenames'},
            {'rank': 32, 'id': 175914, 'name': 'Gloomhaven: Jaws of the Lion'},
            {'rank': 33, 'id': 115746, 'name': 'War of the Ring: Second Edition'},
            {'rank': 34, 'id': 129622, 'name': 'Love Letter'},
            {'rank': 35, 'id': 70323, 'name': 'King of Tokyo'},
            {'rank': 36, 'id': 54043, 'name': 'Shogun'},
            {'rank': 37, 'id': 42215, 'name': 'Dungeon Lords'},
            {'rank': 38, 'id': 25613, 'name': 'In the Year of the Dragon'},
            {'rank': 39, 'id': 41114, 'name': 'Tobago'},
            {'rank': 40, 'id': 38453, 'name': 'Small World'},
            {'rank': 41, 'id': 37380, 'name': 'Castle Ravenloft Board Game'},
            {'rank': 42, 'id': 24480, 'name': 'Shadows over Camelot'},
            {'rank': 43, 'id': 40692, 'name': 'Small World Underground'},
            {'rank': 44, 'id': 91, 'name': 'Cosmic Encounter'},
            {'rank': 45, 'id': 34635, 'name': 'Stone Age'},
            {'rank': 46, 'id': 40398, 'name': 'Lord of Waterdeep'},
            {'rank': 47, 'id': 117959, 'name': 'Orléans'},
            {'rank': 48, 'id': 155426, 'name': 'Inis'},
            {'rank': 49, 'id': 133038, 'name': 'Robinson Crusoe: Adventures on the Cursed Island'},
            {'rank': 50, 'id': 150376, 'name': 'Eldritch Horror'},
        ]
        
        # Erweitere auf TARGET_TOP_GAMES
        extended_games = base_games.copy()
        
        # Weitere bekannte Spiele hinzufügen
        additional_known_games = [
            {'rank': 51, 'id': 124361, 'name': 'Istanbul'},
            {'rank': 52, 'id': 126042, 'name': 'Elysium'},
            {'rank': 53, 'id': 171623, 'name': 'The Castles of Burgundy'},
            {'rank': 54, 'id': 72125, 'name': 'Eclipse'},
            {'rank': 55, 'id': 65244, 'name': 'Forbidden Island'},
            {'rank': 56, 'id': 85325, 'name': 'Alien Frontiers'},
            {'rank': 57, 'id': 42297, 'name': 'Android: Netrunner'},
            {'rank': 58, 'id': 9209, 'name': 'Ticket to Ride: Europe'},
            {'rank': 59, 'id': 18602, 'name': 'Caylus'},
            {'rank': 60, 'id': 9216, 'name': 'Puerto Rico'},
        ]
        
        extended_games.extend(additional_known_games)
        
        # Fülle mit generierten Spielen auf bis TARGET_TOP_GAMES erreicht ist
        while len(extended_games) < TARGET_TOP_GAMES:
            rank = len(extended_games) + 1
            game_id = 500000 + rank  # Hohe IDs um Kollisionen zu vermeiden (unabhängig von TARGET_TOP_GAMES)
            
            extended_games.append({
                'rank': rank,
                'id': game_id,
                'name': f'Top Game #{rank}'
            })
        
        # Auf Zielanzahl begrenzen
        final_games = extended_games[:TARGET_TOP_GAMES]
        
        print(f"✓ {len(final_games)} Fallback-Spiele bereitgestellt")
        
        # Speichere Fallback
        self.save_top_games_cache(final_games, len(final_games), len(final_games))
        
        return final_games
    
    def save_top_games_cache(self, games, total_scraped, total_unique):
        """Speichert Top-Spiele Cache mit Metadaten"""
        metadata = {
            'source': 'scraped' if total_scraped != total_unique else 'fallback',
            'stats': {
                'total_scraped': total_scraped,
                'total_unique': total_unique,
                'duplicates_removed': total_scraped - total_unique,
                'final_count': len(games)
            }
        }
        
        self.cache.save_json_cache('top_games', games, metadata, 'games')
    
    def load_top_games(self):
        """Lädt Top-Spiele und stellt sicher, dass TARGET_TOP_GAMES eindeutige verfügbar sind"""
        cache_result = self.cache.load_json_cache('top_games', 'games')
        
        if cache_result:
            games, metadata = cache_result
            cache_path = self.cache._get_cache_path('top_games', 'json', 'games')
            should_update = self.cache.should_update_cache(cache_path)
            
            if not should_update:
                print(f"📁 Top {TARGET_TOP_GAMES} Cache gefunden (weniger als {self.cache.max_age_days} Tage alt)")
                update_choice = self.ask_user_update_choice(f"Neue Top {TARGET_TOP_GAMES}")
            else:
                print(f"📁 Top {TARGET_TOP_GAMES} Cache gefunden (älter als {self.cache.max_age_days} Tage)")
                update_choice = self.ask_user_update_choice(f"Aktualisierte Top {TARGET_TOP_GAMES}")
        else:
            print(f"📁 Kein Top {TARGET_TOP_GAMES} Cache gefunden")
            update_choice = True
        
        if update_choice:
            top_games = self.scrape_bgg_top_games()
        else:
            print(f"📖 Lade Top {TARGET_TOP_GAMES} aus lokalem Cache...")
            top_games = games
            cache_time = metadata.get('timestamp', 'Unbekannt')
            stats = metadata.get('metadata', {}).get('stats', {})
            
            print(f"✓ {len(top_games)} Spiele aus Cache geladen (erstellt: {cache_time})")
            if stats and stats.get('duplicates_removed', 0) > 0:
                print(f"  Cache-Stats: {stats['duplicates_removed']} Duplikate entfernt von {stats['total_scraped']} ursprünglichen Spielen")
        
        # Finale Validierung - stelle sicher, dass genug Spiele vorhanden sind
        if len(top_games) < TARGET_TOP_GAMES:
            print(f"⚠️  Nur {len(top_games)} Spiele verfügbar, erweitere auf {TARGET_TOP_GAMES}...")
            missing_count = TARGET_TOP_GAMES - len(top_games)
            additional_games = self.generate_additional_games(len(top_games), missing_count)
            top_games.extend(additional_games)
            
            # Speichere erweiterte Liste
            self.save_top_games_cache(top_games, len(top_games), len(top_games))
        
        # Prüfe auf Duplikate ein letztes Mal
        print("🔍 Finale Duplikat-Prüfung...")
        final_games = self.remove_duplicates_from_games(top_games)
        
        return final_games[:TARGET_TOP_GAMES]
    
    def generate_additional_games(self, current_count, needed_count):
        """Generiert zusätzliche Spiele falls benötigt"""
        additional_games = []
        
        print(f"🎲 Generiere {needed_count} zusätzliche Spiele...")
        
        for i in range(needed_count):
            rank = current_count + i + 1
            game_id = 600000 + rank  # Hohe IDs um Kollisionen zu vermeiden
            
            additional_games.append({
                'rank': rank,
                'id': game_id,
                'name': f'Additional Game #{rank}'
            })
        
        return additional_games
    
    def fetch_user_collection(self, username):
        """Lädt die Brettspielsammlung des Nutzers mit Caching"""
        cache_result = self.cache.load_user_cache(username, 'collection')
        
        if cache_result:
            games, metadata = cache_result
            timestamp = metadata.get('timestamp')
            
            # Prüfe Cache-Alter anhand des Timestamps
            if timestamp:
                cache_time = datetime.fromisoformat(timestamp)
                age = datetime.now() - cache_time
                should_update = age > timedelta(days=self.cache.max_age_days)
            else:
                should_update = True
            
            if not should_update:
                print(f"📁 Sammlung für {username} im Cache gefunden (weniger als {self.cache.max_age_days} Tage alt)")
                update_choice = self.ask_user_update_choice("Neue Sammlung")
            else:
                print(f"📁 Sammlung für {username} im Cache gefunden (älter als {self.cache.max_age_days} Tage)")
                update_choice = self.ask_user_update_choice("Aktualisierte Sammlung")
        else:
            print(f"📁 Keine Sammlung für {username} im Cache gefunden")
            update_choice = True
        
        if update_choice:
            games = self._fetch_user_collection_from_api(username)
            if games is not None:
                self.cache.save_user_cache(username, 'collection', games, {'total_games': len(games)})
            return games
        else:
            print(f"📖 Lade Sammlung für {username} aus lokalem Cache...")
            return cache_result[0] if cache_result else None
    
    def _fetch_user_collection_from_api(self, username):
        """Lädt die Brettspielsammlung des Nutzers von der BGG API"""
        url = f"{BGG_API_BASE_URL}/collection?username={username}&stats=1"
        
        print(f"🌐 Lade Sammlung für {username} von BGG...")
        response = requests.get(url)
        
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            games = []
            
            for item in root.findall('item'):
                game_data = {
                    'id': int(item.get('objectid')),
                    'name': item.find('name').text if item.find('name') is not None else 'Unknown',
                    'rating': None,
                    'owned': item.get('subtype') == 'boardgame'
                }
                
                # Bewertung extrahieren
                rating_elem = item.find('.//rating')
                if rating_elem is not None and rating_elem.get('value') != 'N/A':
                    try:
                        game_data['rating'] = float(rating_elem.get('value'))
                    except:
                        pass
                
                games.append(game_data)
            
            print(f"✓ {len(games)} Spiele in der Sammlung gefunden")
            return games
        else:
            print(f"❌ Fehler beim Laden der Sammlung: {response.status_code}")
            return None
    
    
    def fetch_user_plays(self, username, pages=10):
        """Lädt die Spielstatistiken des Nutzers mit Caching"""
        cache_result = self.cache.load_user_cache(username, 'plays')
        
        if cache_result:
            plays, metadata = cache_result
            timestamp = metadata.get('timestamp')
            
            # Prüfe Cache-Alter anhand des Timestamps
            if timestamp:
                cache_time = datetime.fromisoformat(timestamp)
                age = datetime.now() - cache_time
                should_update = age > timedelta(days=self.cache.max_age_days)
            else:
                should_update = True
            
            if not should_update:
                print(f"📁 Spielstatistiken für {username} im Cache gefunden (weniger als {self.cache.max_age_days} Tage alt)")
                update_choice = self.ask_user_update_choice("Neue Spielstatistiken")
            else:
                print(f"📁 Spielstatistiken für {username} im Cache gefunden (älter als {self.cache.max_age_days} Tage)")
                update_choice = self.ask_user_update_choice("Aktualisierte Spielstatistiken")
        else:
            print(f"📁 Keine Spielstatistiken für {username} im Cache gefunden")
            update_choice = True
        
        if update_choice:
            plays = self._fetch_user_plays_from_api(username, pages)
            if plays is not None:
                self.cache.save_user_cache(username, 'plays', plays, {'total_plays': len(plays), 'pages': pages})
            return plays
        else:
            print(f"📖 Lade Spielstatistiken für {username} aus lokalem Cache...")
            return cache_result[0] if cache_result else None
    
    def _fetch_user_plays_from_api(self, username, pages=10):
        """Lädt die Spielstatistiken des Nutzers von der BGG API"""
        all_plays = []
        
        print(f"🌐 Lade Spielstatistiken für {username} von BGG...")
        
        for page in range(1, pages + 1):
            url = f"{BGG_API_BASE_URL}/plays?username={username}&page={page}"
            response = requests.get(url)
            
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                plays = root.findall('play')
                
                if not plays:
                    break
                
                for play in plays:
                    item = play.find('item')
                    if item is not None:
                        all_plays.append({
                            'game_id': int(item.get('objectid')),
                            'game_name': item.get('name'),
                            'date': play.get('date'),
                            'quantity': int(play.get('quantity', 1))
                        })
                
                print(f"  Seite {page}: {len(plays)} Einträge")
                time.sleep(API_DELAY)
            else:
                break
        
        if all_plays:
            print(f"✓ {len(all_plays)} Spieleinträge gefunden")
            return all_plays
        else:
            print("⚠️  Keine Spielstatistiken gefunden")
            return None
    
    
    def fetch_game_details(self, game_ids):
        """Lädt detaillierte Informationen für Spiele"""
        if not game_ids:
            return {}
        
        print(f"🔍 Lade Spieldetails für {len(game_ids)} Spiele...")
        
        game_details = {}
        processed = 0
        
        for i in range(0, len(game_ids), BATCH_SIZE):
            batch = game_ids[i:i+BATCH_SIZE]
            ids_str = ','.join(map(str, batch))
            
            url = f"{BGG_API_BASE_URL}/thing?id={ids_str}&stats=1"
            response = requests.get(url)
            
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                
                for item in root.findall('item'):
                    game_id = int(item.get('id'))
                    
                    # Grunddaten
                    name_elem = item.find('name[@type="primary"]')
                    name = name_elem.get('value') if name_elem is not None else 'Unknown'
                    
                    # Kategorien und Mechaniken
                    categories = [link.get('value') for link in item.findall('link[@type="boardgamecategory"]')]
                    mechanics = [link.get('value') for link in item.findall('link[@type="boardgamemechanic"]')]
                    
                    # Autoren und Illustratoren
                    designers = [link.get('value') for link in item.findall('link[@type="boardgamedesigner"]')]
                    artists = [link.get('value') for link in item.findall('link[@type="boardgameartist"]')]
                    publishers = [link.get('value') for link in item.findall('link[@type="boardgamepublisher"]')]
                    
                    # Spieler und Zeit
                    min_players = self._extract_int_value(item.find('minplayers'), 2)
                    max_players = self._extract_int_value(item.find('maxplayers'), 4)
                    playing_time = self._extract_int_value(item.find('playingtime'), 60)
                    year_published = self._extract_int_value(item.find('yearpublished'), 2000)
                    
                    # Statistiken
                    stats = item.find('statistics/ratings')
                    avg_rating = 0
                    complexity = 0
                    
                    if stats is not None:
                        avg_elem = stats.find('average')
                        if avg_elem is not None:
                            try:
                                avg_rating = float(avg_elem.get('value'))
                            except:
                                pass
                        
                        complex_elem = stats.find('averageweight')
                        if complex_elem is not None:
                            try:
                                complexity = float(complex_elem.get('value'))
                            except:
                                pass
                    
                    game_details[game_id] = {
                        'name': name,
                        'categories': categories,
                        'mechanics': mechanics,
                        'designers': designers,
                        'artists': artists,
                        'publishers': publishers,
                        'year_published': year_published,
                        'avg_rating': avg_rating,
                        'complexity': complexity,
                        'min_players': min_players,
                        'max_players': max_players,
                        'playing_time': playing_time
                    }
            
            processed += len(batch)
            time.sleep(API_DELAY)
            
            if processed % SHOW_PROGRESS_EVERY == 0 or processed >= len(game_ids):
                print(f"  {processed}/{len(game_ids)} Spiele verarbeitet")
        
        return game_details
    
    def _extract_int_value(self, elem, default):
        """Hilfsfunktion zum sicheren Extrahieren von Integer-Werten"""
        if elem is not None:
            try:
                return int(elem.get('value'))
            except:
                pass
        return default
    
    def save_game_details_cache(self, game_details):
        """Speichert Spieldetails im Cache"""
        metadata = {
            'total_games': len(game_details),
            'categories_count': len(set().union(*[details.get('categories', []) for details in game_details.values()])),
            'mechanics_count': len(set().union(*[details.get('mechanics', []) for details in game_details.values()]))
        }
        
        self.cache.save_json_cache('game_details', game_details, metadata, 'games')
    
    def load_game_details_cache(self):
        """Lädt Spieldetails aus dem Cache"""
        cache_result = self.cache.load_json_cache('game_details', 'games')
        
        if cache_result:
            game_details, metadata = cache_result
            print(f"📖 {len(game_details)} Spieldetails aus Cache geladen")
            return game_details
        else:
            print("📁 Kein Game Details Cache gefunden")
            return {}