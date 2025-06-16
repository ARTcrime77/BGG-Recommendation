"""
Cache Manager für das BGG ML-Empfehlungssystem

Zentrale Verwaltung aller Caching-Operationen mit einheitlicher API
und erweiterten Features für ML-Model-Persistierung.
"""

import os
import json
import pickle
import hashlib
import gzip
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np

from config import CACHE_DIR, CACHE_MAX_AGE_DAYS


class CacheManager:
    """
    Zentrale Cache-Verwaltung für das BGG Empfehlungssystem
    
    Verwaltet verschiedene Cache-Typen:
    - JSON-Daten (Spielinformationen, Nutzerdaten)
    - ML-Modelle (scikit-learn Modelle, Feature-Matrices)
    - Berechnete Vektoren (Nutzer-Präferenzen)
    - Visualisierungen (Plot-Dateien)
    """
    
    def __init__(self, cache_dir: str = CACHE_DIR, max_age_days: int = CACHE_MAX_AGE_DAYS):
        """
        Initialisiert den Cache Manager
        
        Args:
            cache_dir: Basis-Verzeichnis für Cache-Dateien
            max_age_days: Standard-Maximalalter für Cache-Einträge (Tage)
        """
        self.cache_dir = cache_dir
        self.max_age_days = max_age_days
        self.ensure_cache_dir()
        
        # Cache-Typ-spezifische Konfiguration
        self.cache_types = {
            'json': {'extension': '.json', 'compress': False},
            'pickle': {'extension': '.pkl', 'compress': True},
            'numpy': {'extension': '.npy', 'compress': True},
            'model': {'extension': '.joblib', 'compress': True},
            'plot': {'extension': '.png', 'compress': False}
        }
    
    def ensure_cache_dir(self):
        """Erstellt das Cache-Verzeichnis falls es nicht existiert"""
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Erstelle Unterverzeichnisse für verschiedene Cache-Typen
        subdirs = ['models', 'features', 'plots', 'users', 'games']
        for subdir in subdirs:
            os.makedirs(os.path.join(self.cache_dir, subdir), exist_ok=True)
    
    def _get_cache_path(self, cache_key: str, cache_type: str = 'json', subdir: str = '') -> str:
        """
        Generiert den vollständigen Pfad für einen Cache-Schlüssel
        
        Args:
            cache_key: Eindeutiger Schlüssel für den Cache-Eintrag
            cache_type: Typ des Caches (json, pickle, numpy, model, plot)
            subdir: Unterverzeichnis innerhalb des Cache-Verzeichnisses
            
        Returns:
            Vollständiger Dateipfad für den Cache-Eintrag
        """
        extension = self.cache_types.get(cache_type, {}).get('extension', '.cache')
        
        # Bereinige Cache-Key von problematischen Zeichen
        clean_key = "".join(c for c in cache_key if c.isalnum() or c in ('-', '_', '.'))
        filename = f"{clean_key}{extension}"
        
        if subdir:
            return os.path.join(self.cache_dir, subdir, filename)
        else:
            return os.path.join(self.cache_dir, filename)
    
    def _generate_cache_key(self, *args, **kwargs) -> str:
        """
        Generiert einen eindeutigen Cache-Schlüssel basierend auf Parametern
        
        Args:
            *args: Positionsargumente für Key-Generierung
            **kwargs: Keyword-Argumente für Key-Generierung
            
        Returns:
            MD5-Hash als Cache-Schlüssel
        """
        # Erstelle String aus allen Parametern
        key_parts = []
        for arg in args:
            if isinstance(arg, (list, dict)):
                key_parts.append(json.dumps(arg, sort_keys=True))
            else:
                key_parts.append(str(arg))
        
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (list, dict)):
                key_parts.append(f"{k}:{json.dumps(v, sort_keys=True)}")
            else:
                key_parts.append(f"{k}:{v}")
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def should_update_cache(self, cache_path: str, max_age_days: Optional[int] = None) -> bool:
        """
        Prüft ob ein Cache-Eintrag aktualisiert werden sollte
        
        Args:
            cache_path: Pfad zur Cache-Datei
            max_age_days: Maximalalter in Tagen (None = Standard verwenden)
            
        Returns:
            True wenn Cache aktualisiert werden sollte
        """
        if not os.path.exists(cache_path):
            return True
        
        max_age = max_age_days if max_age_days is not None else self.max_age_days
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        age = datetime.now() - file_time
        
        return age > timedelta(days=max_age)
    
    def ask_user_update_choice(self, cache_description: str, cache_path: str = None) -> bool:
        """
        Fragt den Nutzer ob Cache aktualisiert werden soll
        
        Args:
            cache_description: Beschreibung des Cache-Inhalts
            cache_path: Pfad zur Cache-Datei (für Altersinfo)
            
        Returns:
            True wenn Nutzer Cache aktualisieren möchte
        """
        age_info = ""
        if cache_path and os.path.exists(cache_path):
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
            age = datetime.now() - file_time
            age_info = f" (Alter: {age.days} Tage)"
        
        while True:
            choice = input(f"💭 {cache_description} laden?{age_info} (j/n): ").lower().strip()
            if choice in ['j', 'ja', 'y', 'yes']:
                return True
            elif choice in ['n', 'nein', 'no']:
                return False
            else:
                print("   Bitte 'j' für Ja oder 'n' für Nein eingeben.")
    
    def save_json_cache(self, cache_key: str, data: Any, metadata: Optional[Dict] = None, 
                       subdir: str = '', max_age_days: Optional[int] = None) -> str:
        """
        Speichert Daten als JSON-Cache
        
        Args:
            cache_key: Eindeutiger Cache-Schlüssel
            data: Zu speichernde Daten
            metadata: Zusätzliche Metadaten
            subdir: Unterverzeichnis
            max_age_days: Maximalalter für diesen Cache
            
        Returns:
            Pfad zur gespeicherten Cache-Datei
        """
        cache_path = self._get_cache_path(cache_key, 'json', subdir)
        
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'cache_key': cache_key,
            'max_age_days': max_age_days or self.max_age_days,
            'data': data
        }
        
        if metadata:
            cache_data['metadata'] = metadata
        
        # Konvertiere NumPy-Typen für JSON-Serialisierung
        cache_data = self._convert_numpy_types(cache_data)
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Cache gespeichert: {cache_key}")
        return cache_path
    
    def load_json_cache(self, cache_key: str, subdir: str = '') -> Optional[Tuple[Any, Dict]]:
        """
        Lädt Daten aus JSON-Cache
        
        Args:
            cache_key: Cache-Schlüssel
            subdir: Unterverzeichnis
            
        Returns:
            Tuple aus (Daten, Metadaten) oder None wenn nicht gefunden
        """
        cache_path = self._get_cache_path(cache_key, 'json', subdir)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            data = cache_data.get('data')
            metadata = {
                'timestamp': cache_data.get('timestamp'),
                'cache_key': cache_data.get('cache_key'),
                'max_age_days': cache_data.get('max_age_days'),
                'metadata': cache_data.get('metadata', {})
            }
            
            return data, metadata
            
        except Exception as e:
            print(f"⚠️ Fehler beim Laden von Cache {cache_key}: {e}")
            return None
    
    def save_model_cache(self, cache_key: str, model: Any, feature_matrix: Optional[np.ndarray] = None,
                        feature_names: Optional[List[str]] = None, scaler: Any = None,
                        metadata: Optional[Dict] = None) -> str:
        """
        Speichert ML-Modell und zugehörige Daten
        
        Args:
            cache_key: Cache-Schlüssel
            model: Trainiertes ML-Modell
            feature_matrix: Feature-Matrix
            feature_names: Namen der Features
            scaler: Trained Scaler
            metadata: Zusätzliche Metadaten
            
        Returns:
            Pfad zur gespeicherten Cache-Datei
        """
        cache_path = self._get_cache_path(cache_key, 'pickle', 'models')
        
        model_data = {
            'timestamp': datetime.now().isoformat(),
            'cache_key': cache_key,
            'model': model,
            'feature_matrix': feature_matrix,
            'feature_names': feature_names,
            'scaler': scaler,
            'metadata': metadata or {}
        }
        
        # Komprimierte Speicherung für große ML-Daten
        with gzip.open(cache_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 ML-Modell Cache gespeichert: {cache_key}")
        return cache_path
    
    def load_model_cache(self, cache_key: str) -> Optional[Dict]:
        """
        Lädt ML-Modell und zugehörige Daten
        
        Args:
            cache_key: Cache-Schlüssel
            
        Returns:
            Dictionary mit model, feature_matrix, feature_names, scaler, metadata
        """
        cache_path = self._get_cache_path(cache_key, 'pickle', 'models')
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with gzip.open(cache_path, 'rb') as f:
                model_data = pickle.load(f)
            
            print(f"📁 ML-Modell Cache geladen: {cache_key}")
            return model_data
            
        except Exception as e:
            print(f"⚠️ Fehler beim Laden von ML-Modell Cache {cache_key}: {e}")
            return None
    
    def save_feature_cache(self, cache_key: str, features: np.ndarray, 
                          feature_info: Dict, metadata: Optional[Dict] = None) -> str:
        """
        Speichert berechnete Features
        
        Args:
            cache_key: Cache-Schlüssel
            features: Feature-Matrix
            feature_info: Feature-Informationen
            metadata: Zusätzliche Metadaten
            
        Returns:
            Pfad zur gespeicherten Cache-Datei
        """
        cache_path = self._get_cache_path(cache_key, 'numpy', 'features')
        
        feature_data = {
            'timestamp': datetime.now().isoformat(),
            'cache_key': cache_key,
            'features': features,
            'feature_info': feature_info,
            'metadata': metadata or {}
        }
        
        with gzip.open(cache_path, 'wb') as f:
            pickle.dump(feature_data, f)
        
        print(f"💾 Feature Cache gespeichert: {cache_key}")
        return cache_path
    
    def load_feature_cache(self, cache_key: str) -> Optional[Dict]:
        """
        Lädt berechnete Features
        
        Args:
            cache_key: Cache-Schlüssel
            
        Returns:
            Dictionary mit features, feature_info, metadata
        """
        cache_path = self._get_cache_path(cache_key, 'numpy', 'features')
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with gzip.open(cache_path, 'rb') as f:
                feature_data = pickle.load(f)
            
            print(f"📁 Feature Cache geladen: {cache_key}")
            return feature_data
            
        except Exception as e:
            print(f"⚠️ Fehler beim Laden von Feature Cache {cache_key}: {e}")
            return None
    
    def save_user_cache(self, username: str, cache_type: str, data: Any, 
                       metadata: Optional[Dict] = None) -> str:
        """
        Speichert nutzerspezifische Daten
        
        Args:
            username: BGG Nutzername
            cache_type: Typ der Daten (collection, plays, preferences)
            data: Zu speichernde Daten
            metadata: Zusätzliche Metadaten
            
        Returns:
            Pfad zur gespeicherten Cache-Datei
        """
        cache_key = f"{username}_{cache_type}"
        return self.save_json_cache(cache_key, data, metadata, 'users')
    
    def load_user_cache(self, username: str, cache_type: str) -> Optional[Tuple[Any, Dict]]:
        """
        Lädt nutzerspezifische Daten
        
        Args:
            username: BGG Nutzername
            cache_type: Typ der Daten (collection, plays, preferences)
            
        Returns:
            Tuple aus (Daten, Metadaten) oder None
        """
        cache_key = f"{username}_{cache_type}"
        return self.load_json_cache(cache_key, 'users')
    
    def invalidate_cache(self, cache_key: str, cache_type: str = 'json', subdir: str = '') -> bool:
        """
        Löscht einen Cache-Eintrag
        
        Args:
            cache_key: Cache-Schlüssel
            cache_type: Cache-Typ
            subdir: Unterverzeichnis
            
        Returns:
            True wenn erfolgreich gelöscht
        """
        cache_path = self._get_cache_path(cache_key, cache_type, subdir)
        
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                print(f"🗑️ Cache gelöscht: {cache_key}")
                return True
            except Exception as e:
                print(f"⚠️ Fehler beim Löschen von Cache {cache_key}: {e}")
                return False
        
        return True
    
    def get_cache_info(self, cache_key: str = None, subdir: str = '') -> Dict:
        """
        Gibt Informationen über Cache-Einträge zurück
        
        Args:
            cache_key: Spezifischer Cache-Schlüssel (None = alle)
            subdir: Unterverzeichnis
            
        Returns:
            Dictionary mit Cache-Informationen
        """
        if cache_key:
            # Info für spezifischen Cache
            cache_paths = []
            for cache_type in self.cache_types:
                path = self._get_cache_path(cache_key, cache_type, subdir)
                if os.path.exists(path):
                    cache_paths.append(path)
            
            if not cache_paths:
                return {'exists': False}
            
            info = {'exists': True, 'files': []}
            for path in cache_paths:
                stat = os.stat(path)
                info['files'].append({
                    'path': path,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'age_days': (datetime.now() - datetime.fromtimestamp(stat.st_mtime)).days
                })
            
            return info
        else:
            # Info für alle Caches
            cache_dir_path = os.path.join(self.cache_dir, subdir) if subdir else self.cache_dir
            
            if not os.path.exists(cache_dir_path):
                return {'total_files': 0, 'total_size': 0}
            
            total_files = 0
            total_size = 0
            file_types = {}
            
            for root, dirs, files in os.walk(cache_dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    size = os.path.getsize(file_path)
                    
                    total_files += 1
                    total_size += size
                    
                    ext = os.path.splitext(file)[1]
                    file_types[ext] = file_types.get(ext, 0) + 1
            
            return {
                'total_files': total_files,
                'total_size': total_size,
                'total_size_mb': round(total_size / 1024 / 1024, 2),
                'file_types': file_types
            }
    
    def cleanup_old_cache(self, max_age_days: Optional[int] = None) -> int:
        """
        Löscht alte Cache-Einträge
        
        Args:
            max_age_days: Maximalalter (None = Standard verwenden)
            
        Returns:
            Anzahl gelöschter Dateien
        """
        max_age = max_age_days if max_age_days is not None else self.max_age_days
        cutoff_time = datetime.now() - timedelta(days=max_age)
        
        deleted_count = 0
        
        for root, dirs, files in os.walk(self.cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if file_time < cutoff_time:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        print(f"🗑️ Alte Cache-Datei gelöscht: {file}")
                    except Exception as e:
                        print(f"⚠️ Fehler beim Löschen von {file}: {e}")
        
        if deleted_count > 0:
            print(f"✅ {deleted_count} alte Cache-Dateien gelöscht")
        
        return deleted_count
    
    def _convert_numpy_types(self, obj: Any) -> Any:
        """
        Konvertiert NumPy-Datentypen zu Python-Standard-Datentypen für JSON
        
        Args:
            obj: Zu konvertierendes Objekt
            
        Returns:
            Konvertiertes Objekt
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj


# Globale Cache-Manager Instanz
cache_manager = CacheManager()