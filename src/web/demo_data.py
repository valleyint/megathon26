"""Demo data, preset queries, and Category DAG environment for CommonsCat AI."""

from typing import Dict, Any, List, Tuple
from src.data.category_vocab import CategoryVocabulary
from src.stage2.graph.category_dag import CategoryDAG
from src.common.types import ScoredCandidate


def build_full_commons_environment() -> Tuple[CategoryVocabulary, CategoryDAG]:
    """Builds a rich Wikimedia Commons Category Vocabulary and Category DAG."""
    vocab = CategoryVocabulary()

    # Domain 1: Architectural Hierarchy (Gothic / French / Cathedrals)
    cat_arch = vocab.add_category("Architecture")
    cat_gothic = vocab.add_category("Gothic architecture")
    cat_rayonnant = vocab.add_category("Rayonnant Gothic architecture")
    cat_cathedral = vocab.add_category("Cathedrals in France")
    cat_notredame = vocab.add_category("Notre-Dame de Paris")
    cat_notredame_facade = vocab.add_category("Western facade of Notre-Dame de Paris")

    # Domain 2: Towers & Monuments in France
    cat_monuments_fr = vocab.add_category("Monuments and memorials in France")
    cat_tower = vocab.add_category("Towers in France")
    cat_eiffel = vocab.add_category("Eiffel Tower")
    cat_eiffel_night = vocab.add_category("Eiffel Tower at night")
    cat_eiffel_illum = vocab.add_category("Illuminations of the Eiffel Tower")
    cat_louvre = vocab.add_category("Louvre Museum")
    cat_louvre_pyramid = vocab.add_category("Louvre Pyramid")

    # Domain 3: Bridges & UK Monuments (Spatial Distractors)
    cat_bridges = vocab.add_category("Bridges by country")
    cat_bridges_uk = vocab.add_category("Bridges in the United Kingdom")
    cat_bridges_london = vocab.add_category("Bridges in London")
    cat_tower_bridge = vocab.add_category("Tower Bridge")
    cat_tower_bridge_sunset = vocab.add_category("Tower Bridge at sunset")
    cat_big_ben = vocab.add_category("Big Ben in London")

    # Domain 4: Italian Monuments (Spatial Distractors)
    cat_ancient_rome = vocab.add_category("Ancient Roman architecture")
    cat_amphi_italy = vocab.add_category("Amphitheatres in Italy")
    cat_colosseum = vocab.add_category("Colosseum in Rome")
    cat_colosseum_ext = vocab.add_category("Exterior of the Colosseum")

    # Domain 5: Indian Monuments
    cat_monuments_india = vocab.add_category("Monuments and memorials in India")
    cat_taj = vocab.add_category("Taj Mahal")
    cat_taj_sunrise = vocab.add_category("Taj Mahal at sunrise")

    # Domain 6: Events & Demonstrations (Temporal / Event Discovery)
    cat_old_protest = vocab.add_category("Demonstrations in Paris in 2015")
    cat_event_2023 = vocab.add_category("Demonstrations and protests in Paris in 2023")
    cat_olympics_2024 = vocab.add_category("2024 Summer Olympics in Paris")
    cat_olympics_torch = vocab.add_category("Torch relay of the 2024 Summer Olympics")
    cat_protests_london_2021 = vocab.add_category("Demonstrations in London in 2021")

    # Domain 7: Transport
    cat_transport_fr = vocab.add_category("Rail transport in France")
    cat_tgv = vocab.add_category("TGV trains in France")

    # Instantiate Category DAG
    dag = CategoryDAG(vocab=vocab)
    for info in vocab.id_to_cat.values():
        dag.add_category(info.category_id, info.name)

    # Hierarchical links: Architecture
    dag.add_edge(cat_arch, cat_gothic)
    dag.add_edge(cat_gothic, cat_rayonnant)
    dag.add_edge(cat_gothic, cat_cathedral)
    dag.add_edge(cat_cathedral, cat_notredame)
    dag.add_edge(cat_notredame, cat_notredame_facade)

    # Hierarchical links: Towers & Paris
    dag.add_edge(cat_monuments_fr, cat_tower)
    dag.add_edge(cat_monuments_fr, cat_louvre)
    dag.add_edge(cat_louvre, cat_louvre_pyramid)
    dag.add_edge(cat_tower, cat_eiffel)
    dag.add_edge(cat_eiffel, cat_eiffel_night)
    dag.add_edge(cat_eiffel_night, cat_eiffel_illum)

    # Hierarchical links: UK Bridges
    dag.add_edge(cat_bridges, cat_bridges_uk)
    dag.add_edge(cat_bridges_uk, cat_bridges_london)
    dag.add_edge(cat_bridges_london, cat_tower_bridge)
    dag.add_edge(cat_tower_bridge, cat_tower_bridge_sunset)

    # Hierarchical links: Rome
    dag.add_edge(cat_arch, cat_ancient_rome)
    dag.add_edge(cat_ancient_rome, cat_amphi_italy)
    dag.add_edge(cat_amphi_italy, cat_colosseum)
    dag.add_edge(cat_colosseum, cat_colosseum_ext)

    # Hierarchical links: India
    dag.add_edge(cat_monuments_india, cat_taj)
    dag.add_edge(cat_taj, cat_taj_sunrise)

    # Hierarchical links: Events
    dag.add_edge(cat_olympics_2024, cat_olympics_torch)

    # Hierarchical links: Transport
    dag.add_edge(cat_transport_fr, cat_tgv)

    return vocab, dag


# Inline SVG presets for fast, self-contained rendering with zero external image dependencies
PRESET_SVGS = {
    "eiffel_protest": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300" width="100%" height="100%">
  <defs>
    <linearGradient id="sky_night" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#0a0f24" />
      <stop offset="60%" stop-color="#141f3d" />
      <stop offset="100%" stop-color="#2a2542" />
    </linearGradient>
    <radialGradient id="illum" cx="50%" cy="40%" r="50%">
      <stop offset="0%" stop-color="#ffdd77" stop-opacity="0.8"/>
      <stop offset="100%" stop-color="#ffaa22" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="tower_gold" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#fff8d6" />
      <stop offset="50%" stop-color="#f5ba42" />
      <stop offset="100%" stop-color="#c97c12" />
    </linearGradient>
  </defs>
  <rect width="400" height="300" fill="url(#sky_night)"/>
  <circle cx="200" cy="110" r="110" fill="url(#illum)"/>
  <!-- Eiffel Tower Silhoutte and Beams -->
  <line x1="200" y1="20" x2="360" y2="0" stroke="#fff8b0" stroke-width="2" opacity="0.6"/>
  <line x1="200" y1="20" x2="40" y2="0" stroke="#fff8b0" stroke-width="2" opacity="0.6"/>
  <polygon points="196,25 204,25 208,120 192,120" fill="url(#tower_gold)"/>
  <rect x="188" y="120" width="24" height="10" rx="3" fill="#ffcc44"/>
  <polygon points="186,130 214,130 224,210 176,210" fill="url(#tower_gold)"/>
  <path d="M 180 210 Q 200 170 220 210 Z" fill="#0a0f24"/>
  <polygon points="172,210 228,210 236,250 164,250" fill="url(#tower_gold)"/>
  <!-- Street & Crowd Protesters -->
  <rect x="0" y="245" width="400" height="55" fill="#111622"/>
  <g fill="#e27d60" opacity="0.85">
    <circle cx="60" cy="255" r="4"/><rect x="57" y="259" width="6" height="15" rx="2"/>
    <circle cx="95" cy="252" r="4"/><rect x="92" y="256" width="6" height="18" rx="2"/>
    <circle cx="130" cy="257" r="4"/><rect x="127" y="261" width="6" height="16" rx="2"/>
    <circle cx="160" cy="250" r="4"/><rect x="157" y="254" width="6" height="20" rx="2"/>
    <circle cx="200" cy="253" r="5"/><rect x="196" y="258" width="8" height="18" rx="2"/>
    <circle cx="240" cy="251" r="4"/><rect x="237" y="255" width="6" height="19" rx="2"/>
    <circle cx="280" cy="256" r="4"/><rect x="277" y="260" width="6" height="16" rx="2"/>
    <circle cx="320" cy="252" r="4"/><rect x="317" y="256" width="6" height="18" rx="2"/>
    <circle cx="355" cy="254" r="4"/><rect x="352" y="258" width="6" height="17" rx="2"/>
  </g>
  <!-- Protest Banners / Flares -->
  <rect x="145" y="242" width="45" height="10" rx="2" fill="#e63946"/>
  <circle cx="200" cy="245" r="7" fill="#ff5722" opacity="0.9"/>
  <circle cx="200" cy="245" r="14" fill="#ff9800" opacity="0.3"/>
  <text x="20" y="35" fill="#ffffff" font-family="sans-serif" font-size="12" font-weight="bold" opacity="0.9">Eiffel Tower at Night · 2023 Protests</text>
</svg>""",

    "notre_dame": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300" width="100%" height="100%">
  <defs>
    <linearGradient id="sky_day" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#4a709c" />
      <stop offset="60%" stop-color="#8ba9c9" />
      <stop offset="100%" stop-color="#d4e2f0" />
    </linearGradient>
    <linearGradient id="stone" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#c8bca7" />
      <stop offset="50%" stop-color="#ded4c3" />
      <stop offset="100%" stop-color="#b8ab94" />
    </linearGradient>
  </defs>
  <rect width="400" height="300" fill="url(#sky_day)"/>
  <!-- Notre-Dame Gothic Cathedral Towers -->
  <rect x="120" y="60" width="45" height="160" fill="url(#stone)"/>
  <rect x="235" y="60" width="45" height="160" fill="url(#stone)"/>
  <rect x="165" y="110" width="70" height="110" fill="url(#stone)"/>
  <!-- Central Rose Window -->
  <circle cx="200" cy="150" r="24" fill="#446688" stroke="#334455" stroke-width="3"/>
  <circle cx="200" cy="150" r="16" fill="none" stroke="#ded4c3" stroke-width="2"/>
  <circle cx="200" cy="150" r="8" fill="#bb3333"/>
  <!-- Gothic Portals -->
  <path d="M 130 220 L 130 255 L 160 255 L 160 220 Q 145 205 130 220 Z" fill="#2d3748"/>
  <path d="M 180 210 L 180 255 L 220 255 L 220 210 Q 200 190 180 210 Z" fill="#2d3748"/>
  <path d="M 240 220 L 240 255 L 270 255 L 270 220 Q 255 205 240 220 Z" fill="#2d3748"/>
  <rect x="0" y="255" width="400" height="45" fill="#5a6578"/>
  <text x="20" y="35" fill="#ffffff" font-family="sans-serif" font-size="12" font-weight="bold">Western Facade · Cathédrale Notre-Dame</text>
</svg>""",

    "tower_bridge": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300" width="100%" height="100%">
  <defs>
    <linearGradient id="sunset" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#2d1b4e" />
      <stop offset="40%" stop-color="#8f3b60" />
      <stop offset="70%" stop-color="#d96b43" />
      <stop offset="100%" stop-color="#f4a261" />
    </linearGradient>
    <linearGradient id="water" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#1f3b5c" />
      <stop offset="100%" stop-color="#0e1a2b" />
    </linearGradient>
  </defs>
  <rect width="400" height="300" fill="url(#sunset)"/>
  <!-- River Thames -->
  <rect x="0" y="220" width="400" height="80" fill="url(#water)"/>
  <!-- Bridge Towers -->
  <rect x="110" y="90" width="40" height="130" fill="#3a4756"/>
  <polygon points="105,90 130,45 155,90" fill="#2c3540"/>
  <rect x="250" y="90" width="40" height="130" fill="#3a4756"/>
  <polygon points="245,90 270,45 295,90" fill="#2c3540"/>
  <!-- Upper Walkway & Suspension Cords -->
  <rect x="145" y="105" width="110" height="12" fill="#526173"/>
  <rect x="145" y="125" width="110" height="8" fill="#526173"/>
  <path d="M 0 170 Q 60 180 110 170" fill="none" stroke="#526173" stroke-width="4"/>
  <path d="M 290 170 Q 340 180 400 170" fill="none" stroke="#526173" stroke-width="4"/>
  <!-- Roadway -->
  <rect x="0" y="195" width="400" height="15" fill="#1b2533"/>
  <text x="20" y="35" fill="#ffffff" font-family="sans-serif" font-size="12" font-weight="bold">Tower Bridge at Sunset · London</text>
</svg>""",

    "colosseum": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300" width="100%" height="100%">
  <defs>
    <linearGradient id="rome_sky" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#3a7bd5" />
      <stop offset="100%" stop-color="#3a6073" />
    </linearGradient>
    <linearGradient id="travertine" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#d4af37" />
      <stop offset="50%" stop-color="#edd59e" />
      <stop offset="100%" stop-color="#b58d2a" />
    </linearGradient>
  </defs>
  <rect width="400" height="300" fill="url(#rome_sky)"/>
  <!-- Colosseum Arcades -->
  <ellipse cx="200" cy="190" rx="170" ry="80" fill="url(#travertine)"/>
  <ellipse cx="200" cy="180" rx="150" ry="65" fill="#4a3e2a"/>
  <!-- Outer Wall Arcades -->
  <path d="M 40 190 Q 200 130 360 190 L 360 140 Q 200 80 40 140 Z" fill="#cfab59"/>
  <!-- Arches Tier 1 -->
  <g fill="#2e261a">
    <path d="M 70 160 Q 80 140 90 160 Z" />
    <path d="M 110 152 Q 120 132 130 152 Z" />
    <path d="M 150 148 Q 160 128 170 148 Z" />
    <path d="M 190 145 Q 200 125 210 145 Z" />
    <path d="M 230 148 Q 240 128 250 148 Z" />
    <path d="M 270 152 Q 280 132 290 152 Z" />
    <path d="M 310 160 Q 320 140 330 160 Z" />
  </g>
  <rect x="0" y="240" width="400" height="60" fill="#2d4a22"/>
  <text x="20" y="35" fill="#ffffff" font-family="sans-serif" font-size="12" font-weight="bold">Exterior of the Colosseum · Rome</text>
</svg>""",

    "olympics_paris": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300" width="100%" height="100%">
  <defs>
    <linearGradient id="paris_festive" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#14213d" />
      <stop offset="50%" stop-color="#003566" />
      <stop offset="100%" stop-color="#001d3d" />
    </linearGradient>
  </defs>
  <rect width="400" height="300" fill="url(#paris_festive)"/>
  <!-- Olympic Rings Style Silhouette -->
  <circle cx="120" cy="110" r="35" fill="none" stroke="#0085C7" stroke-width="8"/>
  <circle cx="200" cy="110" r="35" fill="none" stroke="#F4C300" stroke-width="8"/>
  <circle cx="280" cy="110" r="35" fill="none" stroke="#009F3D" stroke-width="8"/>
  <circle cx="160" cy="150" r="35" fill="none" stroke="#FFFFFF" stroke-width="8"/>
  <circle cx="240" cy="150" r="35" fill="none" stroke="#DF0024" stroke-width="8"/>
  <!-- Torch Relay Flame -->
  <polygon points="190,210 210,210 205,270 195,270" fill="#e5e5e5"/>
  <path d="M 190 210 Q 180 180 200 170 Q 220 180 210 210 Z" fill="#ff7b00"/>
  <path d="M 195 205 Q 190 188 200 180 Q 210 188 205 205 Z" fill="#ffd000"/>
  <text x="20" y="35" fill="#ffffff" font-family="sans-serif" font-size="12" font-weight="bold">Paris 2024 Olympic Games · Torch Relay</text>
</svg>""",

    "taj_mahal": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300" width="100%" height="100%">
  <defs>
    <linearGradient id="sunrise_sky" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#f3904f" />
      <stop offset="50%" stop-color="#3b4371" />
      <stop offset="100%" stop-color="#fdfbfb" />
    </linearGradient>
    <linearGradient id="marble" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#fff5eb" />
      <stop offset="50%" stop-color="#ffffff" />
      <stop offset="100%" stop-color="#f7e6d7" />
    </linearGradient>
  </defs>
  <rect width="400" height="300" fill="url(#sunrise_sky)"/>
  <!-- Reflection Pool -->
  <rect x="0" y="235" width="400" height="65" fill="#1e3c72"/>
  <!-- Taj Dome & Minarets -->
  <rect x="60" y="100" width="10" height="135" fill="url(#marble)"/>
  <rect x="330" y="100" width="10" height="135" fill="url(#marble)"/>
  <!-- Main Tomb Structure -->
  <rect x="130" y="130" width="140" height="105" fill="url(#marble)"/>
  <path d="M 170 130 Q 200 70 230 130 Z" fill="url(#marble)"/>
  <!-- Archway -->
  <path d="M 175 180 L 175 235 L 225 235 L 225 180 Q 200 160 175 180 Z" fill="#301934"/>
  <text x="20" y="35" fill="#ffffff" font-family="sans-serif" font-size="12" font-weight="bold">Taj Mahal at Sunrise · Agra</text>
</svg>"""
}


DEMO_PRESETS = [
    {
        "id": "eiffel_protest_2023",
        "title": "Eiffel Tower Protests (Paris, 2023)",
        "badge": "Night + Protests + Paris",
        "description": "Complex multi-stage case demonstrating spatial pruning (London/Rome), temporal pruning (2015), and DAG ancestor suppression (Eiffel Tower).",
        "svg_key": "eiffel_protest",
        "query": {
            "title": "File:Night_view_of_Eiffel_Tower_during_protests_2023.jpg",
            "description": "{{Information|description=Protests and demonstrations in Paris at night near Champ de Mars with illuminated Eiffel tower}}",
            "timestamp": "2023:05:14 21:30:00",
            "latitude": 48.8584,
            "longitude": 2.2945,
        },
        "stage1_simulation": [
            {"name": "Category:Eiffel Tower at night", "score": 0.9250, "source": "knn", "exemplars": ["hist_fr_102.jpg", "hist_fr_44.jpg"]},
            {"name": "Category:Eiffel Tower", "score": 0.8840, "source": "prototype", "exemplars": []},
            {"name": "Category:Towers in France", "score": 0.8120, "source": "knn", "exemplars": ["hist_fr_09.jpg"]},
            {"name": "Category:Bridges in London", "score": 0.6800, "source": "knn", "exemplars": ["hist_uk_21.jpg"]},
            {"name": "Category:Demonstrations in Paris in 2015", "score": 0.6550, "source": "prototype", "exemplars": []},
            {"name": "Category:Colosseum in Rome", "score": 0.5200, "source": "prototype", "exemplars": []},
            {"name": "Category:Monuments and memorials in France", "score": 0.7300, "source": "knn", "exemplars": ["hist_fr_81.jpg"]},
        ]
    },
    {
        "id": "notre_dame_2022",
        "title": "Cathédrale Notre-Dame (Paris, 2022)",
        "badge": "Gothic Architecture",
        "description": "Demonstrates structural DAG ancestor suppression: broad Architecture & Gothic categories pruned in favor of specific Cathedral leaf.",
        "svg_key": "notre_dame",
        "query": {
            "title": "File:Cathedrale_Notre-Dame_de_Paris_facade_ouest.jpg",
            "description": "{{Information|description=Western facade of Notre-Dame de Paris cathedral, famous rayonnant gothic architecture in Île de la Cité}}",
            "timestamp": "2022:09:18 14:15:00",
            "latitude": 48.8530,
            "longitude": 2.3499,
        },
        "stage1_simulation": [
            {"name": "Category:Western facade of Notre-Dame de Paris", "score": 0.9410, "source": "knn", "exemplars": ["hist_nd_01.jpg", "hist_nd_12.jpg"]},
            {"name": "Category:Notre-Dame de Paris", "score": 0.9150, "source": "prototype", "exemplars": []},
            {"name": "Category:Cathedrals in France", "score": 0.8720, "source": "knn", "exemplars": ["hist_fr_77.jpg"]},
            {"name": "Category:Rayonnant Gothic architecture", "score": 0.8350, "source": "prototype", "exemplars": []},
            {"name": "Category:Gothic architecture", "score": 0.7950, "source": "prototype", "exemplars": []},
            {"name": "Category:Architecture", "score": 0.7100, "source": "knn", "exemplars": ["hist_gen_03.jpg"]},
            {"name": "Category:Tower Bridge", "score": 0.4900, "source": "knn", "exemplars": ["hist_uk_05.jpg"]},
        ]
    },
    {
        "id": "tower_bridge_2021",
        "title": "Tower Bridge Sunset (London, 2021)",
        "badge": "Spatial UK Grounding",
        "description": "Geotags in London trigger spatial pruning of all Paris/Rome candidates while preserving specific Thames bridge hierarchies.",
        "svg_key": "tower_bridge",
        "query": {
            "title": "File:London_Tower_Bridge_sunset_view.jpg",
            "description": "{{Information|description=View of iconic Tower Bridge in London over river Thames during golden hour sunset}}",
            "timestamp": "2021:08:12 20:05:00",
            "latitude": 51.5055,
            "longitude": -0.0754,
        },
        "stage1_simulation": [
            {"name": "Category:Tower Bridge at sunset", "score": 0.9320, "source": "knn", "exemplars": ["hist_tb_45.jpg"]},
            {"name": "Category:Tower Bridge", "score": 0.8950, "source": "prototype", "exemplars": []},
            {"name": "Category:Bridges in London", "score": 0.8500, "source": "knn", "exemplars": ["hist_uk_33.jpg"]},
            {"name": "Category:Bridges in the United Kingdom", "score": 0.7800, "source": "knn", "exemplars": ["hist_uk_01.jpg"]},
            {"name": "Category:Eiffel Tower at night", "score": 0.6200, "source": "knn", "exemplars": ["hist_fr_102.jpg"]},
            {"name": "Category:Cathedrals in France", "score": 0.5100, "source": "prototype", "exemplars": []},
        ]
    },
    {
        "id": "colosseum_rome_2019",
        "title": "Colosseum Amphitheatre (Rome, 2019)",
        "badge": "Italian Antiquity",
        "description": "Reverse geocodes to Rome, Italy. Ancestor DAG prunes general ancient Roman architecture in favor of Exterior of the Colosseum.",
        "svg_key": "colosseum",
        "query": {
            "title": "File:Colosseum_in_Rome_amphitheatre_exterior.jpg",
            "description": "{{Information|description=Exterior facade view of Colosseum Flavian amphitheatre in historic central Rome Italy}}",
            "timestamp": "2019:04:10 11:20:00",
            "latitude": 41.8902,
            "longitude": 12.4922,
        },
        "stage1_simulation": [
            {"name": "Category:Exterior of the Colosseum", "score": 0.9380, "source": "knn", "exemplars": ["hist_it_88.jpg"]},
            {"name": "Category:Colosseum in Rome", "score": 0.9020, "source": "prototype", "exemplars": []},
            {"name": "Category:Amphitheatres in Italy", "score": 0.8250, "source": "knn", "exemplars": ["hist_it_14.jpg"]},
            {"name": "Category:Ancient Roman architecture", "score": 0.7600, "source": "prototype", "exemplars": []},
            {"name": "Category:Architecture", "score": 0.6900, "source": "prototype", "exemplars": []},
            {"name": "Category:Bridges in London", "score": 0.4400, "source": "knn", "exemplars": ["hist_uk_33.jpg"]},
        ]
    },
    {
        "id": "paris_olympics_2024",
        "title": "Paris 2024 Olympic Games (Paris, 2024)",
        "badge": "Proactive Event Discovery",
        "description": "Lexical BM25 and temporal matching synthesizes and ranks proactive event categories: 2024 Summer Olympics in Paris & Torch relay.",
        "svg_key": "olympics_paris",
        "query": {
            "title": "File:Paris_2024_Olympic_Games_torch_relay.jpg",
            "description": "{{Information|description=Torch relay celebrations and sporting ceremonies during 2024 Summer Olympics in Paris France}}",
            "timestamp": "2024:07:26 18:45:00",
            "latitude": 48.8566,
            "longitude": 2.3522,
        },
        "stage1_simulation": [
            {"name": "Category:Torch relay of the 2024 Summer Olympics", "score": 0.8750, "source": "knn", "exemplars": ["hist_oly_02.jpg"]},
            {"name": "Category:Monuments and memorials in France", "score": 0.7200, "source": "prototype", "exemplars": []},
            {"name": "Category:Demonstrations in Paris in 2015", "score": 0.6100, "source": "knn", "exemplars": ["hist_fr_15.jpg"]},
            {"name": "Category:Bridges in London", "score": 0.4800, "source": "prototype", "exemplars": []},
        ]
    },
    {
        "id": "taj_mahal_2023",
        "title": "Taj Mahal Sunrise (Agra, 2023)",
        "badge": "Mughal Monument",
        "description": "GPS lat/lon resolves to Agra, Uttar Pradesh, India. Prunes all non-Indian candidates and preserves specific sunrise category.",
        "svg_key": "taj_mahal",
        "query": {
            "title": "File:Taj_Mahal_at_sunrise_Agra_India.jpg",
            "description": "{{Information|description=Taj Mahal white marble mausoleum at sunrise in Agra Uttar Pradesh India}}",
            "timestamp": "2023:11:05 06:15:00",
            "latitude": 27.1751,
            "longitude": 78.0421,
        },
        "stage1_simulation": [
            {"name": "Category:Taj Mahal at sunrise", "score": 0.9520, "source": "knn", "exemplars": ["hist_in_01.jpg", "hist_in_02.jpg"]},
            {"name": "Category:Taj Mahal", "score": 0.9100, "source": "prototype", "exemplars": []},
            {"name": "Category:Monuments and memorials in India", "score": 0.8350, "source": "knn", "exemplars": ["hist_in_10.jpg"]},
            {"name": "Category:Eiffel Tower at night", "score": 0.4300, "source": "knn", "exemplars": ["hist_fr_102.jpg"]},
            {"name": "Category:Colosseum in Rome", "score": 0.3900, "source": "prototype", "exemplars": []},
        ]
    }
]
