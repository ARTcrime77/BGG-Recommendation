"""
Test fixtures and mock data for BGG ML Recommendation System tests
"""

import json
from datetime import datetime

# Mock BGG Collection Data
MOCK_COLLECTION_DATA = [
    {
        'id': 174430,
        'name': 'Gloomhaven',
        'rating': 8.5,
        'owned': True
    },
    {
        'id': 169786,
        'name': 'Scythe', 
        'rating': 7.8,
        'owned': True
    },
    {
        'id': 167791,
        'name': 'Terraforming Mars',
        'rating': 9.0,
        'owned': True
    }
]

# Mock Plays Data
MOCK_PLAYS_DATA = [
    {
        'game_id': 174430,
        'game_name': 'Gloomhaven',
        'date': '2024-01-15',
        'quantity': 1
    },
    {
        'game_id': 169786,
        'game_name': 'Scythe',
        'date': '2024-01-20',
        'quantity': 2
    },
    {
        'game_id': 167791,
        'game_name': 'Terraforming Mars',
        'date': '2024-02-01',
        'quantity': 3
    }
]

# Mock Game Details
MOCK_GAME_DETAILS = {
    174430: {
        'name': 'Gloomhaven',
        'categories': ['Adventure', 'Exploration', 'Fantasy'],
        'mechanics': ['Action Point Allowance System', 'Card Drafting', 'Cooperative Play'],
        'designers': ['Isaac Childres'],
        'artists': ['Alexandr Elichev', 'Josh T. McDowell'],
        'publishers': ['Cephalofair Games'],
        'year_published': 2017,
        'avg_rating': 8.8,
        'complexity': 3.9,
        'min_players': 1,
        'max_players': 4,
        'playing_time': 120
    },
    169786: {
        'name': 'Scythe',
        'categories': ['Economic', 'Fighting', 'Science Fiction'],
        'mechanics': ['Area Control', 'Variable Player Powers', 'Worker Placement'],
        'designers': ['Jamey Stegmaier'],
        'artists': ['Jakub Rozalski'],
        'publishers': ['Stonemaier Games'],
        'year_published': 2016,
        'avg_rating': 8.3,
        'complexity': 3.4,
        'min_players': 1,
        'max_players': 5,
        'playing_time': 115
    },
    167791: {
        'name': 'Terraforming Mars',
        'categories': ['Economic', 'Environmental', 'Science Fiction'],
        'mechanics': ['Card Drafting', 'Hand Management', 'Tile Placement'],
        'designers': ['Jacob Fryxelius'],
        'artists': ['Isaac Fryxelius'],
        'publishers': ['FryxGames'],
        'year_published': 2016,
        'avg_rating': 8.4,
        'complexity': 3.2,
        'min_players': 1,
        'max_players': 5,
        'playing_time': 120
    }
}

# Mock Top Games Data
MOCK_TOP_GAMES = [
    {'rank': 1, 'id': 174430, 'name': 'Gloomhaven'},
    {'rank': 2, 'id': 233078, 'name': 'Twilight Imperium: Fourth Edition'},
    {'rank': 3, 'id': 167791, 'name': 'Terraforming Mars'},
    {'rank': 4, 'id': 220308, 'name': 'Gaia Project'},
    {'rank': 5, 'id': 173346, 'name': '7 Wonders Duel'}
]

# Mock BGG XML API Responses
MOCK_COLLECTION_XML = '''<?xml version="1.0" encoding="utf-8"?>
<items totalitems="3" termsofuse="https://boardgamegeek.com/xmlapi/termsofuse" pubdate="Wed, 19 Mar 2014 21:11:08 +0000">
    <item objecttype="thing" objectid="174430" subtype="boardgame" collid="12345">
        <name sortindex="1">Gloomhaven</name>
        <status own="1" prevowned="0" fortrade="0" want="0" wanttoplay="0" wanttobuy="0" wishlist="0" preordered="0" lastmodified="2024-01-01 12:00:00"/>
        <rating value="8.5"/>
    </item>
    <item objecttype="thing" objectid="169786" subtype="boardgame" collid="12346">
        <name sortindex="1">Scythe</name>
        <status own="1" prevowned="0" fortrade="0" want="0" wanttoplay="0" wanttobuy="0" wishlist="0" preordered="0" lastmodified="2024-01-01 12:00:00"/>
        <rating value="7.8"/>
    </item>
</items>'''

MOCK_GAME_DETAILS_XML = '''<?xml version="1.0" encoding="utf-8"?>
<items termsofuse="https://boardgamegeek.com/xmlapi/termsofuse">
    <item type="boardgame" id="174430">
        <thumbnail>https://cf.geekdo-images.com/sVmNaE7nZ6SSLYUwEKpVNQ__thumb/img/7xvKgKHGbFjLcHqGjJyaEqYrhBE=/fit-in/200x150/filters:strip_icc()/pic2437871.jpg</thumbnail>
        <image>https://cf.geekdo-images.com/sVmNaE7nZ6SSLYUwEKpVNQ__original/img/vlES-keiIopp1bx8hR5cgWlT2QI=/0x0/filters:format(jpeg)/pic2437871.jpg</image>
        <name type="primary" sortindex="1" value="Gloomhaven"/>
        <description>Gloomhaven is a game of Euro-inspired tactical combat...</description>
        <yearpublished value="2017"/>
        <minplayers value="1"/>
        <maxplayers value="4"/>
        <playingtime value="120"/>
        <link type="boardgamecategory" id="1022" value="Adventure"/>
        <link type="boardgamecategory" id="1020" value="Exploration"/>
        <link type="boardgamecategory" id="1010" value="Fantasy"/>
        <link type="boardgamemechanic" id="2001" value="Action Point Allowance System"/>
        <link type="boardgamedesigner" id="69802" value="Isaac Childres"/>
        <link type="boardgameartist" id="69802" value="Alexandr Elichev"/>
        <statistics page="1">
            <ratings>
                <usersrated value="88574"/>
                <average value="8.77729"/>
                <bayesaverage value="8.59138"/>
                <ranks>
                    <rank type="subtype" id="1" name="boardgame" friendlyname="Board Game Rank" value="1"/>
                </ranks>
                <stddev value="1.50968"/>
                <median value="0"/>
                <owned value="134863"/>
                <trading value="881"/>
                <wanting value="8267"/>
                <wishing value="34195"/>
                <numcomments value="13412"/>
                <numweights value="2972"/>
                <averageweight value="3.86"/>
            </ratings>
        </statistics>
    </item>
</items>'''

MOCK_PLAYS_XML = '''<?xml version="1.0" encoding="utf-8"?>
<plays username="testuser" userid="12345" total="3" page="1" termsofuse="https://boardgamegeek.com/xmlapi/termsofuse">
    <play id="123456" date="2024-01-15" quantity="1" length="0" incomplete="0" nowinstats="1" location="">
        <item name="Gloomhaven" objecttype="thing" objectid="174430"/>
    </play>
    <play id="123457" date="2024-01-20" quantity="2" length="0" incomplete="0" nowinstats="1" location="">
        <item name="Scythe" objecttype="thing" objectid="169786"/>
    </play>
</plays>'''

# Mock Cache Data Structures
MOCK_TOP500_CACHE = {
    'timestamp': datetime.now().isoformat(),
    'games': MOCK_TOP_GAMES,
    'source': 'test',
    'stats': {
        'total_scraped': len(MOCK_TOP_GAMES),
        'total_unique': len(MOCK_TOP_GAMES),
        'duplicates_removed': 0,
        'final_count': len(MOCK_TOP_GAMES)
    }
}

MOCK_GAME_DETAILS_CACHE = {
    'timestamp': datetime.now().isoformat(),
    'details': MOCK_GAME_DETAILS
}

# Mock HTTP Responses
class MockResponse:
    def __init__(self, content, status_code=200):
        self.content = content.encode('utf-8')
        self.status_code = status_code
        self.text = content

def get_mock_collection_response():
    return MockResponse(MOCK_COLLECTION_XML)

def get_mock_game_details_response():
    return MockResponse(MOCK_GAME_DETAILS_XML)

def get_mock_plays_response():
    return MockResponse(MOCK_PLAYS_XML)

def get_mock_scraping_response():
    # Mock HTML response for BGG browse page
    html_content = '''
    <html>
    <body>
    <table>
        <tr id="row_1">
            <td class="collection_rank">1</td>
            <td class="collection_thumbnail">
                <a href="/boardgame/174430/gloomhaven">
                    <img alt="Gloomhaven" src="image.jpg">
                </a>
            </td>
        </tr>
        <tr id="row_2">
            <td class="collection_rank">2</td>
            <td class="collection_thumbnail">
                <a href="/boardgame/233078/twilight-imperium-fourth-edition">
                    <img alt="Twilight Imperium: Fourth Edition" src="image2.jpg">
                </a>
            </td>
        </tr>
    </table>
    </body>
    </html>
    '''
    return MockResponse(html_content)