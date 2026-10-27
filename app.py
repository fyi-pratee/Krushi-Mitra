from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from datetime import datetime
import json
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

app = Flask(__name__)

# --- LOAD ASSETS ---
try:
    model = pickle.load(open('crop_model.pkl', 'rb'))
    location_encoder = pickle.load(open('location_encoder.pkl', 'rb'))
    soil_type_encoder = pickle.load(open('soil_type_encoder.pkl', 'rb'))
    print("✓ Successfully loaded all model files")
except FileNotFoundError as e:
    print("Error: .pkl files not found. Run your training script first!")
    print(f"Missing file: {e}")
    exit(1)
except Exception as e:
    print(f"Error loading model files: {e}")
    exit(1)

# ─── ENRICHED CROP DATABASE ────────────────────────────────────────────────
# Each entry now includes:
#   kannada_name  : local Kannada name
#   npk_ranges    : dict with N/P/K typical soil ranges (kg/ha)
#   ph_range      : optimal soil pH range string
#   moisture_range: optimal volumetric soil moisture %
#   temp_range    : optimal temperature range
#   regions       : key Karnataka districts/regions where grown
#   fertilizer    : recommended fertilizer dose
#   profit        : economic value description
#   desc          : short description
#   img           : image URL
# ──────────────────────────────────────────────────────────────────────────

CROP_INFO = {
    'coconut': {
        'kannada_name': 'ತೆಂಗಿನಕಾಯಿ (Tenginakāyi)',
        'desc': "Coconut is a highly profitable perennial crop providing oil, water, coir and shell products. Karnataka is among India's leading producers with a strong processing industry.",
        'npk_ranges': {'N': '37–73 kg/ha', 'P': '39–71 kg/ha', 'K': '162–238 kg/ha'},
        'ph_range': '5.3 – 6.3 (moderately acidic)',
        'moisture_range': '28 – 42 % (moderate to high, well-irrigated)',
        'temp_range': '25 – 31 °C (warm tropical)',
        'suitable_soil': 'Coastal Alluvial, Laterite, Alluvial/Red',
        'regions': 'Tumakuru, Hassan, Dakshina Kannada, Udupi, Haveri',
        'fertilizer': 'NPK 500:320:1200 g per palm per year; split into 2 doses at onset and end of monsoon. Drip fertigation preferred.',
        'profit': 'High – Long-term steady income from nuts, coconut oil, coir and shell products.',
        'img': 'https://www.biovie.fr/modules/prestablog/views/img/grid-for-1-7/up-img/750.jpg',
    },
    'cotton': {
        'kannada_name': 'ಹತ್ತಿ (Hatti)',
        'desc': "Known as 'White Gold', cotton is the dominant cash crop of North Karnataka's black-soil belt. It thrives in deep Black (Regur) soils with good drainage.",
        'npk_ranges': {'N': '67–123 kg/ha', 'P': '28–56 kg/ha', 'K': '22–54 kg/ha'},
        'ph_range': '7.1 – 8.1 (neutral to mildly alkaline)',
        'moisture_range': '13 – 23 % (low to moderate, drought-tolerant)',
        'temp_range': '25 – 31 °C (warm, dry spells beneficial)',
        'suitable_soil': 'Black, Red/Black mix, Red Sandy',
        'regions': 'Dharwad, Haveri, Gadag, Raichur, Ballari, Koppal, Chitradurga, Davanagere, Bagalkot',
        'fertilizer': 'NPK 120:60:60 kg/ha + FYM 5 t/ha; apply N in 3 splits (basal, 30 DAS, 60 DAS).',
        'profit': 'Moderate to High – Depends on market price and pest management; Bt cotton varieties reduce costs.',
        'img': 'https://cdn.britannica.com/18/156618-050-39339EA2/cotton-harvesting.jpg',
    },
    'coffee': {
        'kannada_name': 'ಕಾಫಿ (Kāphi)',
        'desc': "Coffee is the backbone of the plantation economy in Kodagu and Chikkamagaluru. Karnataka accounts for ~70% of India's coffee production, growing both Arabica and Robusta varieties.",
        'npk_ranges': {'N': '93–137 kg/ha', 'P': '17–33 kg/ha', 'K': '38–66 kg/ha'},
        'ph_range': '5.9 – 6.4 (slightly to moderately acidic)',
        'moisture_range': '29 – 41 % (high humidity, shade canopy)',
        'temp_range': '21 – 25 °C (cool to warm, high altitude 600–1500 m)',
        'suitable_soil': 'Acidic Loam, Loamy Forest Soil, Laterite',
        'regions': 'Kodagu (Coorg), Chikkamagaluru, Hassan, Shivamogga',
        'fertilizer': 'NPK 40:20:40 g per plant + micronutrients; apply before onset of monsoon in 2 splits.',
        'profit': 'High – Significant export value; Arabica fetches premium on specialty markets.',
        'img': 'https://www.aboutcoffee.org/wp-content/uploads/2024/10/ripe-coffee-cherries-on-branch-of-coffee-tree.jpg',
    },
    'sugarcane': {
        'kannada_name': 'ಕಬ್ಬು (Kabbu)',
        'desc': "A key industrial crop used for sugar and ethanol production. Requires high water availability and fertile loamy or black soils. Karnataka has numerous cooperative sugar factories.",
        'npk_ranges': {'N': '123–187 kg/ha', 'P': '46–78 kg/ha', 'K': '87–143 kg/ha'},
        'ph_range': '6.4 – 7.3 (near neutral)',
        'moisture_range': '25 – 39 % (moderate to high, irrigated)',
        'temp_range': '25 – 31 °C (warm, frost-free)',
        'suitable_soil': 'Loamy, Black, Alluvial/Red',
        'regions': 'Mandya, Belagavi, Mysuru, Shivamogga, Hassan',
        'fertilizer': 'NPK 250:100:100 kg/ha + FYM 25 t/ha; apply N in 3 splits through the crop cycle.',
        'profit': 'High – Stable buyback price from sugar mills; intercropping with soybean increases income.',
        'img': 'https://shop.growinnhomes.com/cdn/shop/files/IMG_97032_1.heic?v=1755122933&width=1946',
    },
    'mustard': {
        'kannada_name': 'ಸಾಸಿವೆ (Sāsive)',
        'desc': "An important oilseed Rabi crop grown in North Karnataka during the winter season. Highly tolerant of cool, dry conditions and well-adapted to black and loamy soils.",
        'npk_ranges': {'N': '60–100 kg/ha', 'P': '28–52 kg/ha', 'K': '12–28 kg/ha'},
        'ph_range': '5.8 – 6.7 (slightly acidic)',
        'moisture_range': '13 – 23 % (low to moderate)',
        'temp_range': '19 – 25 °C (cool, Rabi season)',
        'suitable_soil': 'Loamy, Black, Red Loam',
        'regions': 'Kalaburagi, Vijayapura, Bidar, Dharwad, Gadag',
        'fertilizer': 'NPK 80:40:20 kg/ha + sulphur 40 kg/ha; apply half N at sowing, remainder at 30 DAS.',
        'profit': 'Moderate – Low cultivation cost with decent returns; oil cake adds value.',
        'img': 'https://tiimg.tistatic.com/fp/1/008/911/mustard-seeds-386.jpg',
    },
    'soybean': {
        'kannada_name': 'ಸೋಯಾಬೀನ್ (Sōyābīn)',
        'desc': "A dual-purpose Kharif crop providing edible oil and protein-rich animal feed. Being a legume, it fixes atmospheric nitrogen and improves soil health for the following crop.",
        'npk_ranges': {'N': '15–35 kg/ha', 'P': '46–74 kg/ha', 'K': '28–52 kg/ha'},
        'ph_range': '5.8 – 6.8 (slightly acidic)',
        'moisture_range': '15 – 25 % (moderate)',
        'temp_range': '23 – 29 °C (warm, Kharif season)',
        'suitable_soil': 'Black, Loamy, Red/Black mix',
        'regions': 'Belagavi, Bidar, Vijayapura, Dharwad, Gadag',
        'fertilizer': 'NPK 20:60:40 kg/ha; inoculate seeds with Rhizobium culture before sowing.',
        'profit': 'Moderate – Short growing cycle allows crop rotation; nitrogen fixation improves next crop yield.',
        'img': 'https://img.freepik.com/free-photo/soybean-isolated-white-background_74190-7801.jpg?semt=ais_hybrid&w=740&q=80',
    },
    'chillies': {
        'kannada_name': 'ಮೆಣಸಿನಕಾಯಿ (Meṇasinakāyi)',
        'desc': "Famous for the GI-tagged Byadgi variety, Karnataka's chillies are globally exported for their deep red colour and mild heat. North Karnataka is the primary growing belt.",
        'npk_ranges': {'N': '84–136 kg/ha', 'P': '41–69 kg/ha', 'K': '39–71 kg/ha'},
        'ph_range': '6.0 – 7.0 (slightly acidic to neutral)',
        'moisture_range': '16 – 28 % (moderate, well-drained)',
        'temp_range': '24 – 30 °C (warm days, cool nights)',
        'suitable_soil': 'Red Loam, Loamy, Red/Black mix, Black',
        'regions': 'Ballari, Haveri (Byadgi), Raichur, Kalaburagi, Bidar, Dharwad',
        'fertilizer': 'NPK 120:60:60 kg/ha; foliar spray of micronutrients at flowering boosts colour and yield.',
        'profit': 'Very High – Premium export demand for colour-grade Byadgi variety; spice processing adds value.',
        'img': 'https://www.shutterstock.com/image-photo/byadgi-red-dry-chilli-indian-600nw-2005127822.jpg',
    },
    'cardamom': {
        'kannada_name': 'ಏಲಕ್ಕಿ (Ēlakki)',
        'desc': "The 'Queen of Spices' thrives in the evergreen shaded forests of the Western Ghats. Cardamom needs lower nitrogen, higher potassium, and cooler temperatures than coffee.",
        'npk_ranges': {'N': '34–62 kg/ha', 'P': '22–42 kg/ha', 'K': '64–100 kg/ha'},
        'ph_range': '5.1 – 5.6 (moderately acidic)',
        'moisture_range': '32 – 44 % (high, shade-grown)',
        'temp_range': '17 – 22 °C (cool, humid, 600–1500 m altitude)',
        'suitable_soil': 'Loamy Forest Soil, Acidic Loam, Laterite',
        'regions': 'Kodagu, Chikkamagaluru, Hassan, Shivamogga (Western Ghats belt)',
        'fertilizer': 'NPK 75:75:150 kg/ha; apply in 3 splits during monsoon. Compost 5 t/ha at planting.',
        'profit': 'High – One of the most valuable spices by weight; volatile oil content commands premium.',
        'img': 'https://exoticflora.in/cdn/shop/products/8434971801_caf25895ba_b_900x.jpg?v=1633340061',
    },
    'tea': {
        'kannada_name': 'ಚಹಾ (Cahā)',
        'desc': "Grown in the highest-altitude regions of the Western Ghats, Karnataka tea requires the most acidic soils and coolest temperatures of all plantation crops. Processed as CTC and orthodox varieties.",
        'npk_ranges': {'N': '103–147 kg/ha', 'P': '20–36 kg/ha', 'K': '33–57 kg/ha'},
        'ph_range': '4.4 – 4.9 (strongly acidic)',
        'moisture_range': '39 – 51 % (high, consistent rainfall required)',
        'temp_range': '15 – 20 °C (cool, misty highlands above 900 m)',
        'suitable_soil': 'Acidic Loam, Loamy Forest Soil, Laterite',
        'regions': 'Chikkamagaluru (Mudigere, Aldur), Kodagu, Hassan',
        'fertilizer': 'NPK 120:20:50 kg/ha; apply in 4 equal splits annually. Avoid liming — acidic soil is essential.',
        'profit': 'High – Domestic market demand steady; specialty orthodox grades earn export premium.',
        'img': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTH8ihqIbrjnrCdCDEQstYSnzXJS_fXgZQvNA&s',
    },
    'arecanut': {
        'kannada_name': 'ಅಡಿಕೆ (Aḍike)',
        'desc': "A major commercial crop in coastal and hilly Karnataka. Arecanut (betel nut) is deeply embedded in cultural practices and has a vast domestic market. Karnataka is India's top producer.",
        'npk_ranges': {'N': '60–100 kg/ha', 'P': '28–52 kg/ha', 'K': '92–148 kg/ha'},
        'ph_range': '5.6 – 6.5 (slightly to moderately acidic)',
        'moisture_range': '31 – 45 % (high, well-irrigated)',
        'temp_range': '24 – 30 °C (warm humid tropical)',
        'suitable_soil': 'Coastal Alluvial, Laterite, Red Loam',
        'regions': 'Dakshina Kannada, Udupi, Shivamogga, Uttara Kannada, Kodagu',
        'fertilizer': 'NPK 100:40:140 g per palm per year; apply in 2 splits. Drip fertigation preferred.',
        'profit': 'Very High – Karnataka is India\'s #1 arecanut producer; prices highly remunerative.',
        'img': 'https://3.imimg.com/data3/YK/QS/MY-11002287/arecanut.jpg',
    },
    'mulberry': {
        'kannada_name': 'ಹಿಪ್ಪುನೇರಳೆ (Hippunēraḷe)',
        'desc': "The primary food for silkworms. Mulberry cultivation is essential for Karnataka's world-famous silk industry, centred around Ramanagara and Mysuru. Requires the highest nitrogen of all crops.",
        'npk_ranges': {'N': '162–238 kg/ha', 'P': '60–100 kg/ha', 'K': '60–100 kg/ha'},
        'ph_range': '6.1 – 7.0 (slightly acidic to neutral)',
        'moisture_range': '21 – 35 % (moderate to high)',
        'temp_range': '24 – 30 °C (warm, humid)',
        'suitable_soil': 'Loamy, Red Loam, Alluvial/Red',
        'regions': 'Mandya, Ramanagara, Kolar, Mysuru, Hassan, Tumakuru',
        'fertilizer': 'NPK 300:120:120 kg/ha per year; irrigate regularly. Split into 4 doses for continuous leaf supply.',
        'profit': 'Moderate to High – Income linked to cocoon price; integrated with sericulture enterprise.',
        'img': 'https://i5.walmartimages.com/seo/Dwarf-Mulberry-Tree-Plant-1-Gallon_2754fde9-9149-442f-a6d8-fa9a20d13515.b6c0247a32bd3629b7961391ef3362cc.jpeg',
    },
    'ragi': {
        'kannada_name': 'ರಾಗಿ (Rāgi)',
        'desc': "Finger millet (Ragi) is a nutritional superfood and dietary staple in South Karnataka. Highly drought-tolerant and rich in calcium, iron and fibre. Grown on red laterite soils.",
        'npk_ranges': {'N': '40–80 kg/ha', 'P': '20–50 kg/ha', 'K': '18–38 kg/ha'},
        'ph_range': '5.9 – 7.1 (slightly acidic to neutral)',
        'moisture_range': '15 – 29 % (low to moderate, rainfed)',
        'temp_range': '22 – 28 °C (warm, Kharif & Rabi)',
        'suitable_soil': 'Red, Red Loam, Red Sandy, Laterite/Red Loam',
        'regions': 'Tumakuru, Bengaluru Rural, Bengaluru Urban, Kolar, Chikballapur, Ramanagara, Mysuru, Hassan',
        'fertilizer': 'NPK 60:30:30 kg/ha + FYM 5 t/ha; zinc 25 kg/ha and sulphur 20 kg/ha improve grain yield.',
        'profit': 'Moderate – Low input cost; rising demand for health foods boosting farm-gate price.',
        'img': 'https://www.herbyangel.com/cdn/shop/articles/Decoding_the_Health_Benefits_of_Ragi_and_Amaranth_2048x.webp?v=1700732156',
    },
    'turmeric': {
        'kannada_name': 'ಅರಿಶಿನ (Ariśina)',
        'desc': "A medicinal spice with high export demand driven by curcumin's pharmaceutical applications. Grows best in well-drained loamy soils with good organic matter content.",
        'npk_ranges': {'N': '68–112 kg/ha', 'P': '36–64 kg/ha', 'K': '68–112 kg/ha'},
        'ph_range': '5.6 – 6.5 (slightly to moderately acidic)',
        'moisture_range': '28 – 42 % (moderate, well-drained)',
        'temp_range': '23 – 29 °C (warm, humid)',
        'suitable_soil': 'Loamy, Red Loam, Alluvial/Red',
        'regions': 'Chamarajanagar, Mysuru, Hassan, Mandya, Belagavi',
        'fertilizer': 'NPK 60:50:120 kg/ha + FYM 30 t/ha at planting; apply N in 3 splits during crop growth.',
        'profit': 'High – Strong pharmaceutical and export demand; curcumin-rich varieties command premium.',
        'img': 'https://i.herbalreality.com/wp-content/uploads/2023/11/22183558/Fresh-turmeric-root-Curcuma-longa-scaled-600x600.jpg',
    },
}

# Kannada/regional aliases → canonical key
CROP_ALIASES = {
    'tenginakay': 'coconut', 'tenginakāyi': 'coconut', 'ತೆಂಗು': 'coconut',
    'hatti': 'cotton', 'ಹತ್ತಿ': 'cotton',
    'kaphi': 'coffee', 'kāphi': 'coffee', 'ಕಾಫಿ': 'coffee',
    'kabbu': 'sugarcane', 'kab': 'sugarcane', 'ಕಬ್ಬು': 'sugarcane',
    'sasive': 'mustard', 'sāsive': 'mustard', 'ಸಾಸಿವೆ': 'mustard',
    'soyabin': 'soybean', 'sōyābīn': 'soybean', 'ಸೋಯಾ': 'soybean',
    'menasinakay': 'chillies', 'byadgi': 'chillies', 'ಮೆಣಸು': 'chillies',
    'elakki': 'cardamom', 'ēlakki': 'cardamom', 'ಏಲಕ್ಕಿ': 'cardamom',
    'chaha': 'tea', 'cahā': 'tea', 'ಚಹಾ': 'tea',
    'adike': 'arecanut', 'aḍike': 'arecanut', 'betel nut': 'arecanut', 'ಅಡಿಕೆ': 'arecanut',
    'hippunerale': 'mulberry', 'hippunēraḷe': 'mulberry', 'ಹಿಪ್ಪುನೇರಳೆ': 'mulberry',
    'ragi': 'ragi', 'finger millet': 'ragi', 'ರಾಗಿ': 'ragi',
    'arishina': 'turmeric', 'ariśina': 'turmeric', 'haldi': 'turmeric', 'ಅರಿಶಿನ': 'turmeric',
}


def resolve_crop_key(name: str):
    """Map any input name (English, Kannada, alias) to CROP_INFO key."""
    key = name.strip().lower()
    # Direct match
    if key in CROP_INFO:
        return key
    # Alias match
    if key in CROP_ALIASES:
        return CROP_ALIASES[key]
    # Partial match
    for k in CROP_INFO:
        if k in key or key in k:
            return k
    return None


# ─── ROUTES ───────────────────────────────────────────────────────────────

@app.route('/')
def home():
    districts = sorted(list(location_encoder.classes_))
    soil_types = sorted(list(soil_type_encoder.classes_))
    return render_template('index.html', districts=districts, soil_types=soil_types)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        loc_name = request.form.get('location')
        if not loc_name:
            return "Please select a district!", 400

        soil_type = request.form.get('soil_type')
        if not soil_type:
            return "Please select a soil type!", 400

        loc_encoded = location_encoder.transform([loc_name])[0]
        soil_encoded = soil_type_encoder.transform([soil_type])[0]

        n        = float(request.form.get('N'))
        p        = float(request.form.get('P'))
        k        = float(request.form.get('K'))
        temp     = float(request.form.get('temp'))
        ph       = float(request.form.get('ph'))
        moisture = float(request.form.get('moisture'))

        input_features = np.array([[n, p, k, temp, ph, moisture, loc_encoded, soil_encoded]])

        # --- Top-3 recommendations ---
        probs   = model.predict_proba(input_features)[0]
        classes = model.classes_

        # Sort by probability descending, take top 3
        top_indices = np.argsort(probs)[::-1][:3]

        top3 = []
        for idx in top_indices:
            crop_name = classes[idx]
            confidence = round(probs[idx] * 100, 2)
            details = CROP_INFO.get(crop_name.lower(), {
                'kannada_name': '',
                'desc': 'No detailed information available for this crop.',
                'npk_ranges': {'N': 'N/A', 'P': 'N/A', 'K': 'N/A'},
                'ph_range': 'N/A',
                'moisture_range': 'N/A',
                'temp_range': 'N/A',
                'suitable_soil': 'N/A',
                'regions': 'Consult local agricultural office.',
                'fertilizer': 'Consult local agricultural office.',
                'profit': 'Unknown',
                'img': '',
            })
            top3.append({
                'crop':       crop_name.capitalize(),
                'confidence': confidence,
                'info':       details,
            })

        inputs = {
            'N': n, 'P': p, 'K': k,
            'temp': temp, 'ph': ph, 'moisture': moisture,
            'soil_type': soil_type,
        }

        return render_template('result.html',
                               mode='recommendation',
                               location=loc_name,
                               top3=top3,
                               inputs=inputs)

    except Exception as e:
        return f"Prediction Error: {str(e)}", 500


@app.route('/crop-info', methods=['POST'])
def crop_info():
    raw = request.form.get('crop_name', '').strip()
    if not raw:
        return "Please enter a crop name.", 400

    key = resolve_crop_key(raw)
    if key is None:
        return render_template('result.html',
                               mode='lookup',
                               crop_name=raw.capitalize(),
                               info={
                                   'kannada_name': '',
                                   'desc': 'No information found for this crop in our Karnataka database. Try the English name or select from the dropdown.',
                                   'npk_ranges': {'N': 'N/A', 'P': 'N/A', 'K': 'N/A'},
                                   'ph_range': 'N/A',
                                   'moisture_range': 'N/A',
                                   'temp_range': 'N/A',
                                   'suitable_soil': 'N/A',
                                   'regions': 'N/A',
                                   'fertilizer': 'Consult local agricultural office.',
                                   'profit': 'Unknown',
                                   'img': '',
                               })

    info = CROP_INFO[key]
    display_name = key.capitalize()
    return render_template('result.html',
                           mode='lookup',
                           crop_name=display_name,
                           info=info)


# ─── OFFLINE PROFIT DATA (Karnataka market rates 2024-25) ─────────────────
# All values per acre. Prices in INR.
PROFIT_DATA = {
    'coconut': {
        'yield_per_acre': '60–80 nuts/palm, ~40 palms/acre',
        'market_price_range': '₹15–₹25 per nut',
        'input_cost': 18000,
        'min_revenue': 36000,
        'max_revenue': 80000,
        'best_season': 'Both',
        'risk_level': 'Low',
        'price_trend': 'Rising',
        'key_tips': [
            'Intercrop with banana or pepper to earn extra ₹15,000–20,000/acre.',
            'Use drip irrigation to cut water cost by 30%.',
            'Sell coconut water and shell separately for 20% more income.',
        ],
        'government_schemes': [
            'PM Coconut Development Board subsidy – up to ₹10,000/acre for replanting.',
            'NABARD plantation crop loan at 7% interest.',
        ],
        'summary': 'Coconut is a reliable long-term earner in coastal Karnataka with low risk. Once established, trees yield for 60+ years with minimal input cost.',
    },
    'cotton': {
        'yield_per_acre': '6–10 quintals (seed cotton)',
        'market_price_range': '₹6,500–₹8,000 per quintal (MSP ₹7,121)',
        'input_cost': 22000,
        'min_revenue': 39000,
        'max_revenue': 80000,
        'best_season': 'Kharif',
        'risk_level': 'Medium',
        'price_trend': 'Stable',
        'key_tips': [
            'Use Bt cotton varieties to reduce pesticide cost by up to ₹5,000/acre.',
            'Sell at APMC after November when prices typically peak.',
            'Intercrop with soybean in early weeks for extra income.',
        ],
        'government_schemes': [
            'PM Fasal Bima Yojana – crop insurance at subsidised premium.',
            'MSP procurement through Karnataka State Co-operative Cotton Growers Society.',
        ],
        'summary': 'Cotton offers solid returns in North Karnataka\'s black soil belt. Market prices above MSP in recent years have made it increasingly attractive.',
    },
    'coffee': {
        'yield_per_acre': '4–8 quintals (clean coffee)',
        'market_price_range': '₹18,000–₹35,000 per quintal',
        'input_cost': 35000,
        'min_revenue': 72000,
        'max_revenue': 280000,
        'best_season': 'Both',
        'risk_level': 'Medium',
        'price_trend': 'Rising',
        'key_tips': [
            'Get Coffee Board registration to access export markets and better prices.',
            'Shade-grown and organic certification can fetch 2x premium.',
            'Arabica yields less but commands ₹10,000+/quintal more than Robusta.',
        ],
        'government_schemes': [
            'Coffee Board of India replanting/rejuvenation subsidy – 50% of cost.',
            'Integrated Coffee Development Project grants for processing equipment.',
        ],
        'summary': 'Coffee is among the highest-value crops in Karnataka, especially in Kodagu and Chikkamagaluru. Global specialty coffee demand is driving prices to record highs.',
    },
    'sugarcane': {
        'yield_per_acre': '35–50 tonnes',
        'market_price_range': '₹3,200–₹3,600 per tonne (FRP)',
        'input_cost': 28000,
        'min_revenue': 112000,
        'max_revenue': 180000,
        'best_season': 'Both',
        'risk_level': 'Low',
        'price_trend': 'Stable',
        'key_tips': [
            'Join a cooperative sugar factory for guaranteed FRP buyback price.',
            'Ratoon cropping (2nd and 3rd crop) cuts input cost by 40%.',
            'Drip irrigation with fertigation can boost yield by 25%.',
        ],
        'government_schemes': [
            'Karnataka Sugarcane Growers Association price support schemes.',
            'PMKSY micro-irrigation subsidy – 55–90% for drip/sprinkler setup.',
        ],
        'summary': 'Sugarcane delivers the highest gross revenue per acre in Karnataka\'s irrigated belt. Guaranteed mill buyback makes cash flow predictable and reliable.',
    },
    'mustard': {
        'yield_per_acre': '5–8 quintals',
        'market_price_range': '₹5,200–₹6,500 per quintal (MSP ₹5,650)',
        'input_cost': 8000,
        'min_revenue': 26000,
        'max_revenue': 52000,
        'best_season': 'Rabi',
        'risk_level': 'Low',
        'price_trend': 'Rising',
        'key_tips': [
            'Sow in October for best yield; delay reduces output significantly.',
            'Oil cake byproduct sells at ₹800–₹1,000/quintal as animal feed.',
            'Zero tillage sowing saves ₹2,000–₹3,000/acre in fuel and labour.',
        ],
        'government_schemes': [
            'National Food Security Mission oilseed component – free seed kits.',
            'MSP procurement through NAFED and Karnataka State Co-ops.',
        ],
        'summary': 'Mustard is the most cost-effective Rabi crop for North Karnataka with very low input needs. Rising edible oil prices domestically are pushing farm-gate prices above MSP.',
    },
    'soybean': {
        'yield_per_acre': '8–12 quintals',
        'market_price_range': '₹4,000–₹5,200 per quintal (MSP ₹4,600)',
        'input_cost': 10000,
        'min_revenue': 32000,
        'max_revenue': 62400,
        'best_season': 'Kharif',
        'risk_level': 'Low',
        'price_trend': 'Stable',
        'key_tips': [
            'Inoculate seeds with Rhizobium to save ₹1,500/acre on nitrogen fertiliser.',
            'Follow soybean with wheat/chickpea to benefit from residual soil nitrogen.',
            'Sell to oil mills directly instead of APMC to get ₹200–₹400 extra per quintal.',
        ],
        'government_schemes': [
            'NMOOP – National Mission on Oilseeds & Oil Palm subsidy on seeds and inputs.',
            'MSP procurement through state-designated agencies.',
        ],
        'summary': 'Soybean is an excellent low-risk Kharif crop that also improves soil for the next season. Short 90-day cycle allows quick turnaround of capital.',
    },
    'chillies': {
        'yield_per_acre': '8–15 quintals (dry)',
        'market_price_range': '₹8,000–₹22,000 per quintal (Byadgi variety)',
        'input_cost': 30000,
        'min_revenue': 64000,
        'max_revenue': 330000,
        'best_season': 'Rabi',
        'risk_level': 'High',
        'price_trend': 'Rising',
        'key_tips': [
            'Grow GI-tagged Byadgi variety for export-grade colour premium.',
            'Sell to spice exporters directly – prices 30–40% above mandi rates.',
            'Use yellow sticky traps to reduce thrips/mite pesticide costs.',
        ],
        'government_schemes': [
            'Spices Board of India export incentive and quality certification support.',
            'PMFME scheme – post-harvest processing unit subsidy up to 35%.',
        ],
        'summary': 'Byadgi chillies offer some of the highest profit potential in Karnataka\'s spice sector. Export demand for colour-grade chillies keeps prices elevated despite production variability.',
    },
    'cardamom': {
        'yield_per_acre': '80–150 kg (cured)',
        'market_price_range': '₹1,200–₹2,500 per kg',
        'input_cost': 40000,
        'min_revenue': 96000,
        'max_revenue': 375000,
        'best_season': 'Both',
        'risk_level': 'Medium',
        'price_trend': 'Rising',
        'key_tips': [
            'Auction at Spices Board e-trading platform for transparent best price.',
            'Maintain shade canopy at 50–60% — excess sun reduces yield sharply.',
            'Drip irrigation during dry spells prevents flower drop.',
        ],
        'government_schemes': [
            'Spices Board replanting/rejuvenation subsidy – 50% for cardamom.',
            'NABARD long-term plantation loan at concessional rates.',
        ],
        'summary': 'Cardamom is among the most lucrative spice crops per acre in Karnataka\'s Western Ghats. Domestic and Middle East demand continues to push prices higher year-on-year.',
    },
    'tea': {
        'yield_per_acre': '400–800 kg (made tea)',
        'market_price_range': '₹120–₹350 per kg',
        'input_cost': 32000,
        'min_revenue': 48000,
        'max_revenue': 280000,
        'best_season': 'Both',
        'risk_level': 'Medium',
        'price_trend': 'Stable',
        'key_tips': [
            'Orthodox processing fetches 2–3x more than CTC for the same leaf.',
            'Direct-to-consumer branding (online/boutique) can get ₹800+/kg.',
            'Regular pruning every 3–4 years maintains yield and quality.',
        ],
        'government_schemes': [
            'Tea Board of India replanting subsidy – ₹1.5 lakh/ha for old bushes.',
            'Quality Upgrade & Product Diversification scheme grants.',
        ],
        'summary': 'Tea from Kodagu and Chikkamagaluru commands premium in specialty markets. While bulk CTC margins are thin, value-added orthodox and single-estate teas are highly profitable.',
    },
    'arecanut': {
        'yield_per_acre': '8–14 quintals (dried)',
        'market_price_range': '₹35,000–₹55,000 per quintal',
        'input_cost': 25000,
        'min_revenue': 280000,
        'max_revenue': 770000,
        'best_season': 'Both',
        'risk_level': 'Low',
        'price_trend': 'Rising',
        'key_tips': [
            'Drip fertigation cuts input cost while boosting nut size and weight.',
            'Store dried nuts 2–3 months post-harvest to sell when prices peak.',
            'Intercrop with pepper on areca poles for ₹30,000–₹50,000 extra/acre.',
        ],
        'government_schemes': [
            'Arecanut & Spices Development Directorate replanting subsidy.',
            'NABARD plantation development loan at 9% p.a.',
        ],
        'summary': 'Arecanut is Karnataka\'s most profitable commercial crop per acre — prices have tripled in the last decade. Low input requirements make ROI exceptionally high for established gardens.',
    },
    'mulberry': {
        'yield_per_acre': '3–5 cocoon crops/year, 150–250 kg cocoons/crop',
        'market_price_range': '₹350–₹600 per kg cocoons',
        'input_cost': 20000,
        'min_revenue': 52500,
        'max_revenue': 150000,
        'best_season': 'Both',
        'risk_level': 'Medium',
        'price_trend': 'Stable',
        'key_tips': [
            'Rear double-hybrid silkworm varieties (PM x CSR) for better cocoon quality.',
            'Maintain chawki rearing centre for first instars to reduce mortality.',
            'Sell directly to reeling units in Ramanagara for ₹50–₹80 extra per kg.',
        ],
        'government_schemes': [
            'Karnataka Silk Industries Corporation input subsidy for sericulture farmers.',
            'Central Silk Board – free disease-free layings and rearing equipment.',
        ],
        'summary': 'Mulberry-based sericulture provides year-round income in 5–6 crops per year. Government support and Karnataka\'s established silk industry make market access reliable.',
    },
    'ragi': {
        'yield_per_acre': '8–14 quintals',
        'market_price_range': '₹3,500–₹5,000 per quintal (MSP ₹3,846)',
        'input_cost': 6000,
        'min_revenue': 28000,
        'max_revenue': 70000,
        'best_season': 'Kharif',
        'risk_level': 'Low',
        'price_trend': 'Rising',
        'key_tips': [
            'Sell as health food (ragi flour, malt) to urban markets at 2–3x mandi price.',
            'Finger millet straw fetches ₹3,000–₹5,000/acre as premium cattle fodder.',
            'Transplanting method instead of broadcasting increases yield by 30%.',
        ],
        'government_schemes': [
            'PMFME scheme – funding for ragi flour milling and packaging unit.',
            'MSP procurement through Karnataka Food & Civil Supplies Corporation.',
        ],
        'summary': 'Ragi has the lowest input cost of all Karnataka crops with near-zero irrigation need. Surging urban health food demand is rapidly pushing prices well above MSP.',
    },
    'turmeric': {
        'yield_per_acre': '25–35 quintals (dry)',
        'market_price_range': '₹7,000–₹15,000 per quintal',
        'input_cost': 22000,
        'min_revenue': 175000,
        'max_revenue': 525000,
        'best_season': 'Kharif',
        'risk_level': 'Medium',
        'price_trend': 'Rising',
        'key_tips': [
            'High-curcumin varieties (Pragati, Sudarsana) fetch 20–30% price premium.',
            'Polish and grade before selling — unpolished turmeric gets ₹2,000 less/quintal.',
            'Export-grade certification through Spices Board opens lucrative overseas markets.',
        ],
        'government_schemes': [
            'Spices Board quality certification and export development assistance.',
            'PMFME – subsidy for turmeric processing and polishing unit setup.',
        ],
        'summary': 'Turmeric offers outstanding returns due to booming pharmaceutical and export demand for curcumin. Prices have surged 60–80% in 2023–24, making it one of Karnataka\'s hottest cash crops.',
    },
}


@app.route('/profit-predict', methods=['POST'])
def profit_predict():
    """Offline rule-based profit prediction using Karnataka market data."""
    try:
        data = request.get_json()
        crop = data.get('crop', '').lower().strip()
        location = data.get('location', 'Karnataka')
        land_acres = float(data.get('land_acres', 1))
        season = data.get('season', 'Kharif')
        confidence = float(data.get('confidence', 0))

        # Resolve crop key
        resolved = resolve_crop_key(crop)
        if not resolved or resolved not in PROFIT_DATA:
            return jsonify({'error': f'No profit data available for "{crop}". Try one of the recommended crops.'}), 404

        d = PROFIT_DATA[resolved]

        # Confidence boosts revenue slightly (higher confidence = better soil match)
        boost = 1.0 + (confidence - 50) * 0.003  # ±15% range
        boost = max(0.85, min(1.15, boost))

        min_rev = int(d['min_revenue'] * boost)
        max_rev = int(d['max_revenue'] * boost)
        input_cost = d['input_cost']

        min_profit = max(0, min_rev - input_cost)
        max_profit = max_rev - input_cost
        avg_profit = (min_profit + max_profit) // 2
        roi = int((avg_profit / input_cost) * 100) if input_cost > 0 else 0

        result = {
            'min_profit_inr': min_profit,
            'max_profit_inr': max_profit,
            'avg_profit_inr': avg_profit,
            'total_profit_min': int(min_profit * land_acres),
            'total_profit_max': int(max_profit * land_acres),
            'yield_per_acre': d['yield_per_acre'],
            'market_price_range': d['market_price_range'],
            'input_cost_per_acre': input_cost,
            'roi_percent': roi,
            'risk_level': d['risk_level'],
            'best_season': d['best_season'],
            'price_trend': d['price_trend'],
            'key_tips': d['key_tips'],
            'government_schemes': d['government_schemes'],
            'summary': d['summary'],
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── SOIL TESTING LABS BY DISTRICT ───────────────────────────────────────
SOIL_LABS = {
    'Bagalkot': [
        {'name': 'KSDA Soil Testing Lab', 'address': 'Dept. of Agriculture, Bagalkot', 'phone': '08354-235180'},
        {'name': 'UAS Dharwad Satellite Lab', 'address': 'KVK Bagalkot, Mudhol Road', 'phone': '08354-230050'},
    ],
    'Ballari': [
        {'name': 'Dept. of Agriculture Soil Lab', 'address': 'Ballari District Agriculture Office', 'phone': '08392-255422'},
        {'name': 'KSSL Bellary Unit', 'address': 'Near APMC Yard, Ballari', 'phone': '08392-278900'},
    ],
    'Belagavi': [
        {'name': 'Soil Testing Laboratory, Belagavi', 'address': 'Dept. of Agriculture, Camp Road, Belagavi', 'phone': '0831-2404213'},
        {'name': 'UAS Dharwad Extension Lab', 'address': 'KVK Belagavi, Udyambag', 'phone': '0831-2463290'},
    ],
    'Bengaluru Rural': [
        {'name': 'GKVK Soil Testing Lab', 'address': 'UAS Bangalore, GKVK Campus, Hebbal', 'phone': '080-23330153'},
        {'name': 'Dept. of Agriculture Soil Lab', 'address': 'Bengaluru Rural District Office, Devanahalli', 'phone': '080-27671208'},
    ],
    'Bengaluru Urban': [
        {'name': 'GKVK Soil Testing Lab', 'address': 'UAS Bangalore, GKVK Campus, Hebbal', 'phone': '080-23330153'},
        {'name': 'Krishi Vigyan Kendra', 'address': 'Doddaballapur Road, Yelahanka', 'phone': '080-28465763'},
    ],
    'Bidar': [
        {'name': 'Soil Testing Lab Bidar', 'address': 'Dept. of Agriculture, Bidar District Office', 'phone': '08482-226541'},
        {'name': 'KVK Bidar', 'address': 'Udgir Road, Bidar', 'phone': '08482-243020'},
    ],
    'Chamarajanagar': [
        {'name': 'Dept. of Agriculture Lab', 'address': 'District Agriculture Office, Chamarajanagar', 'phone': '08226-222061'},
        {'name': 'KVK Chamarajanagar', 'address': 'Yelandur Road, Chamarajanagar', 'phone': '08226-228200'},
    ],
    'Chikballapur': [
        {'name': 'Soil Testing Lab', 'address': 'Dept. of Agriculture, Chikballapur', 'phone': '08156-272040'},
        {'name': 'IIHR Satellite Lab', 'address': 'Hesaraghatta Road, Chikballapur', 'phone': '08156-275010'},
    ],
    'Chikkamagaluru': [
        {'name': 'Coffee Board Soil Lab', 'address': 'Coffee Board Regional Office, Chikkamagaluru', 'phone': '08262-220294'},
        {'name': 'Dept. of Agriculture Lab', 'address': 'District Agriculture Office, Chikkamagaluru', 'phone': '08262-231072'},
    ],
    'Chitradurga': [
        {'name': 'Soil Testing Lab Chitradurga', 'address': 'Dept. of Agriculture, Chitradurga', 'phone': '08194-223093'},
        {'name': 'KVK Chitradurga', 'address': 'Holalkere Road, Chitradurga', 'phone': '08194-232118'},
    ],
    'Dakshina Kannada': [
        {'name': 'Coconut Development Board Lab', 'address': 'CDB Regional Office, Kodialbail, Mangaluru', 'phone': '0824-2440571'},
        {'name': 'Dept. of Agriculture Lab', 'address': 'DK District Office, Mangaluru', 'phone': '0824-2220330'},
    ],
    'Davanagere': [
        {'name': 'Soil Testing Lab Davanagere', 'address': 'Dept. of Agriculture, Davanagere', 'phone': '08192-231530'},
        {'name': 'KVK Davanagere', 'address': 'Harihar Road, Davanagere', 'phone': '08192-245860'},
    ],
    'Dharwad': [
        {'name': 'UAS Dharwad Soil Testing Lab', 'address': 'University of Agricultural Sciences, Dharwad', 'phone': '0836-2447354'},
        {'name': 'KSDA Central Soil Lab', 'address': 'Dept. of Agriculture, PB Road, Dharwad', 'phone': '0836-2441390'},
    ],
    'Gadag': [
        {'name': 'Soil Testing Lab Gadag', 'address': 'Dept. of Agriculture, Gadag District', 'phone': '08372-236120'},
        {'name': 'KVK Gadag', 'address': 'Ron Road, Gadag', 'phone': '08372-248500'},
    ],
    'Hassan': [
        {'name': 'Soil Testing Lab Hassan', 'address': 'Dept. of Agriculture, Hassan', 'phone': '08172-263540'},
        {'name': 'Coffee Board Extension Lab', 'address': 'Coffee Board, Sakleshpur Road, Hassan', 'phone': '08172-268100'},
    ],
    'Haveri': [
        {'name': 'Soil Testing Lab Haveri', 'address': 'Dept. of Agriculture, Haveri', 'phone': '08375-235082'},
        {'name': 'Spices Board Lab', 'address': 'Byadgi Chilli Research Station, Byadgi', 'phone': '08375-248300'},
    ],
    'Kalaburagi': [
        {'name': 'Soil Testing Lab Kalaburagi', 'address': 'Dept. of Agriculture, Kalaburagi', 'phone': '08472-263920'},
        {'name': 'UAS Raichur Extension Lab', 'address': 'KVK Kalaburagi, Afzalpur Road', 'phone': '08472-274050'},
    ],
    'Kodagu': [
        {'name': 'Coffee Board Soil & Plant Tissue Lab', 'address': 'Coffee Board, Madikeri, Kodagu', 'phone': '08272-229752'},
        {'name': 'Dept. of Agriculture Lab', 'address': 'District Agriculture Office, Madikeri', 'phone': '08272-225660'},
    ],
    'Kolar': [
        {'name': 'Soil Testing Lab Kolar', 'address': 'Dept. of Agriculture, Kolar', 'phone': '08152-222063'},
        {'name': 'IIHR Extension Lab', 'address': 'IIHR Regional Station, Mulbagal', 'phone': '08152-272040'},
    ],
    'Koppal': [
        {'name': 'Soil Testing Lab Koppal', 'address': 'Dept. of Agriculture, Koppal', 'phone': '08539-220423'},
        {'name': 'KVK Koppal', 'address': 'Gangavathi Road, Koppal', 'phone': '08539-224000'},
    ],
    'Mandya': [
        {'name': 'Soil Testing Lab Mandya', 'address': 'Dept. of Agriculture, Mandya', 'phone': '08232-222433'},
        {'name': 'SSRI Mandya', 'address': 'Sugarcane Breeding Institute, Mandya', 'phone': '08232-228754'},
    ],
    'Mysuru': [
        {'name': 'Soil Testing Lab Mysuru', 'address': 'Dept. of Agriculture, Nazarbad, Mysuru', 'phone': '0821-2444620'},
        {'name': 'CPCRI Extension Lab', 'address': 'Coconut Research Station, Hirehalli, Tumkur Road', 'phone': '0821-2571010'},
    ],
    'Raichur': [
        {'name': 'UAS Raichur Soil Lab', 'address': 'University of Agricultural Sciences, Raichur', 'phone': '08532-220443'},
        {'name': 'Dept. of Agriculture Lab', 'address': 'District Agriculture Office, Raichur', 'phone': '08532-231088'},
    ],
    'Ramanagara': [
        {'name': 'Sericulture Research Lab', 'address': 'Central Silk Board, Ramanagara', 'phone': '08027-272342'},
        {'name': 'Dept. of Agriculture Lab', 'address': 'District Agriculture Office, Ramanagara', 'phone': '08027-274020'},
    ],
    'Shivamogga': [
        {'name': 'Soil Testing Lab Shivamogga', 'address': 'Dept. of Agriculture, Shivamogga', 'phone': '08182-222620'},
        {'name': 'Arecanut Research Station', 'address': 'CPCRI Vitla Extension Centre, Shivamogga', 'phone': '08182-254300'},
    ],
    'Tumakuru': [
        {'name': 'Soil Testing Lab Tumakuru', 'address': 'Dept. of Agriculture, Tumakuru', 'phone': '0816-2278380'},
        {'name': 'Coconut Research Station', 'address': 'CPCRI Regional Station, Hirehalli, Tumkur', 'phone': '0816-2283440'},
    ],
    'Udupi': [
        {'name': 'Soil Testing Lab Udupi', 'address': 'Dept. of Agriculture, Manipal Road, Udupi', 'phone': '0820-2521032'},
        {'name': 'KVK Udupi', 'address': 'Brahmavar, Udupi', 'phone': '0820-2576200'},
    ],
    'Uttara Kannada': [
        {'name': 'Soil Testing Lab UK', 'address': 'Dept. of Agriculture, Karwar', 'phone': '08382-226033'},
        {'name': 'KVK Uttara Kannada', 'address': 'Sirsi Road, Karwar', 'phone': '08382-236100'},
    ],
    'Vijayanagara': [
        {'name': 'Soil Testing Lab Vijayanagara', 'address': 'Dept. of Agriculture, Hosapete', 'phone': '08394-252100'},
        {'name': 'KVK Vijayanagara', 'address': 'Hosapete-Kampli Road, Hosapete', 'phone': '08394-255200'},
    ],
    'Vijayapura': [
        {'name': 'Soil Testing Lab Vijayapura', 'address': 'Dept. of Agriculture, Vijayapura', 'phone': '08352-250832'},
        {'name': 'UAS Dharwad Extension', 'address': 'KVK Vijayapura, Sholapur Road', 'phone': '08352-264000'},
    ],
    'Yadgir': [
        {'name': 'Soil Testing Lab Yadgir', 'address': 'Dept. of Agriculture, Yadgir', 'phone': '08473-253020'},
        {'name': 'KVK Yadgir', 'address': 'Shorapur Road, Yadgir', 'phone': '08473-255500'},
    ],
}

# ─── SEASONAL INTELLIGENCE ────────────────────────────────────────────────
def get_season_info():
    month = datetime.now().month
    if month in [6, 7, 8, 9, 10]:
        return {
            'season': 'Kharif',
            'period': 'June – October',
            'icon': '🌧️',
            'color': '#2d6a9f',
            'bg': '#e8f4fd',
            'border': '#2d6a9f',
            'message': 'Kharif season is active — ideal time to sow monsoon crops.',
            'recommended_crops': ['Cotton', 'Soybean', 'Ragi', 'Turmeric', 'Chillies'],
            'avoid_crops': ['Mustard', 'Wheat'],
            'tip': 'Ensure field drainage before sowing. Check soil moisture levels after first rains before applying fertiliser.',
            'sowing_window': 'Sow within 2 weeks of monsoon onset (typically mid-June) for best yield.',
        }
    elif month in [11, 12, 1, 2, 3]:
        return {
            'season': 'Rabi',
            'period': 'November – March',
            'icon': '☀️',
            'color': '#8a6010',
            'bg': '#fff8e8',
            'border': '#c8902a',
            'message': 'Rabi season is active — ideal for cool-weather and irrigated crops.',
            'recommended_crops': ['Mustard', 'Chickpea', 'Wheat', 'Chillies', 'Cardamom harvesting'],
            'avoid_crops': ['Cotton', 'Soybean'],
            'tip': 'Soil moisture from retreating monsoon is good. Conduct soil test before applying Rabi fertiliser dose.',
            'sowing_window': 'Sow by first week of November for Rabi crops to avoid terminal heat stress.',
        }
    else:
        return {
            'season': 'Summer / Zaid',
            'period': 'April – May',
            'icon': '🌞',
            'color': '#b54700',
            'bg': '#fff3ee',
            'border': '#e07040',
            'message': 'Summer (Zaid) season — limited crops suitable; focus on perennials and pre-monsoon prep.',
            'recommended_crops': ['Coconut maintenance', 'Arecanut maintenance', 'Coffee pruning'],
            'avoid_crops': ['Ragi', 'Cotton', 'Soybean'],
            'tip': 'This is the best time to get your soil tested so results are ready before Kharif sowing in June.',
            'sowing_window': 'Use this period to prepare land, add organic matter, and correct soil pH before Kharif.',
        }


@app.route('/api/season', methods=['GET'])
def season_api():
    return jsonify(get_season_info())


@app.route('/api/labs/<district>', methods=['GET'])
def labs_api(district):
    labs = SOIL_LABS.get(district, [])
    if not labs:
        # Try partial match
        for key in SOIL_LABS:
            if district.lower() in key.lower() or key.lower() in district.lower():
                labs = SOIL_LABS[key]
                break
    return jsonify({'district': district, 'labs': labs})


@app.route('/analytics')
def analytics():
    """Analytics dashboard with comprehensive visualizations"""
    return render_template('analytics.html')


@app.route('/api/analytics-data', methods=['GET'])
def analytics_data():
    """Generate analytics data for visualizations"""
    try:
        # 1. Soil Nutrient Distribution Data
        crops_list = list(CROP_INFO.keys())
        npk_data = []
        for crop in crops_list:
            info = CROP_INFO[crop]
            npk_ranges = info['npk_ranges']
            # Extract numeric values (take midpoint of ranges)
            n_val = extract_midpoint(npk_ranges['N'])
            p_val = extract_midpoint(npk_ranges['P'])
            k_val = extract_midpoint(npk_ranges['K'])
            npk_data.append({
                'crop': crop.capitalize(),
                'N': n_val,
                'P': p_val,
                'K': k_val
            })
        
        # 2. Environmental Factors Data
        env_data = []
        for crop in crops_list:
            info = CROP_INFO[crop]
            temp_mid = extract_midpoint(info['temp_range'])
            ph_mid = extract_midpoint(info['ph_range'])
            moisture_mid = extract_midpoint(info['moisture_range'])
            env_data.append({
                'crop': crop.capitalize(),
                'temperature': temp_mid,
                'ph': ph_mid,
                'moisture': moisture_mid
            })
        
        # 3. Profitability Analysis Data
        profit_data_list = []
        for crop in crops_list:
            if crop in PROFIT_DATA:
                pdata = PROFIT_DATA[crop]
                avg_profit = (pdata['min_revenue'] - pdata['input_cost'] + 
                             pdata['max_revenue'] - pdata['input_cost']) / 2
                profit_data_list.append({
                    'crop': crop.capitalize(),
                    'min_profit': pdata['min_revenue'] - pdata['input_cost'],
                    'max_profit': pdata['max_revenue'] - pdata['input_cost'],
                    'avg_profit': avg_profit,
                    'input_cost': pdata['input_cost'],
                    'risk_level': pdata['risk_level'],
                    'price_trend': pdata['price_trend']
                })
        
        # 4. Feature Importance (simulated based on crop requirements)
        feature_importance = {
            'Rainfall': 25,
            'K-level': 22,
            'Temperature': 18,
            'N-level': 15,
            'pH': 12,
            'P-level': 8
        }
        
        # 5. Crop Distribution by Season (dynamically generated)
        season_distribution = {'Kharif': [], 'Rabi': [], 'Both': [], 'Summer': []}
        for crop in crops_list:
            if crop in PROFIT_DATA:
                season = PROFIT_DATA[crop]['best_season']
                crop_name = crop.capitalize()
                if season == 'Kharif':
                    season_distribution['Kharif'].append(crop_name)
                elif season == 'Rabi':
                    season_distribution['Rabi'].append(crop_name)
                elif season == 'Summer / Zaid':
                    season_distribution['Summer'].append(crop_name)
                else:  # Both
                    season_distribution['Both'].append(crop_name)
        
        # Remove empty seasons
        season_distribution = {k: v for k, v in season_distribution.items() if v}
        
        # 6. Risk vs Profit Analysis
        risk_profit = []
        risk_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
        for crop in crops_list:
            if crop in PROFIT_DATA:
                pdata = PROFIT_DATA[crop]
                avg_profit = (pdata['min_revenue'] - pdata['input_cost'] + 
                             pdata['max_revenue'] - pdata['input_cost']) / 2
                risk_profit.append({
                    'crop': crop.capitalize(),
                    'risk_score': risk_mapping.get(pdata['risk_level'], 2),
                    'avg_profit': avg_profit,
                    'risk_level': pdata['risk_level']
                })
        
        return jsonify({
            'npk_data': npk_data,
            'env_data': env_data,
            'profit_data': profit_data_list,
            'feature_importance': feature_importance,
            'season_distribution': season_distribution,
            'risk_profit': risk_profit
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def extract_midpoint(range_str):
    """Extract midpoint from range string like '10-40 kg/ha' or '5.5 - 7.0'"""
    try:
        # Remove units and extra text
        range_str = range_str.split('(')[0].strip()
        # Extract numbers
        import re
        numbers = re.findall(r'\d+\.?\d*', range_str)
        if len(numbers) >= 2:
            return (float(numbers[0]) + float(numbers[1])) / 2
        elif len(numbers) == 1:
            return float(numbers[0])
        return 0
    except:
        return 0


@app.route('/api/farmerhelp', methods=['POST'])
def farmerhelp():
    """AI Assistant endpoint for answering farmer queries"""
    try:
        data = request.get_json()
        message = data.get('message', '').lower().strip()
        context = data.get('context', {})
        language = data.get('language', 'en')
        
        # Generate response based on message patterns
        response, suggestions = generate_farmerhelp_response(message, context, language)
        
        return jsonify({
            'response': response,
            'suggestions': suggestions
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def translate_to_kannada(text_dict):
    """Translate response components to Kannada"""
    translations = {
        # Common phrases
        'Current Season': 'ಪ್ರಸ್ತುತ ಋತು',
        'Recommended crops': 'ಶಿಫಾರಸು ಮಾಡಲಾದ ಬೆಳೆಗಳು',
        'Tip': 'ಸಲಹೆ',
        'Profitability Analysis for': 'ಲಾಭದಾಯಕತೆ ವಿಶ್ಲೇಷಣೆ',
        'Expected Profit per Acre': 'ಪ್ರತಿ ಎಕರೆಗೆ ನಿರೀಕ್ಷಿತ ಲಾಭ',
        'Minimum': 'ಕನಿಷ್ಠ',
        'Maximum': 'ಗರಿಷ್ಠ',
        'Input Cost': 'ಇನ್‌ಪುಟ್ ವೆಚ್ಚ',
        'Risk Level': 'ಅಪಾಯ ಮಟ್ಟ',
        'Price Trend': 'ಬೆಲೆ ಪ್ರವೃತ್ತಿ',
        'Top Tip': 'ಪ್ರಮುಖ ಸಲಹೆ',
        'Most Profitable Crops in Karnataka': 'ಕರ್ನಾಟಕದಲ್ಲಿ ಅತ್ಯಂತ ಲಾಭದಾಯಕ ಬೆಳೆಗಳು',
        'Fertilizer Requirements for': 'ಗೊಬ್ಬರ ಅವಶ್ಯಕತೆಗಳು',
        'NPK Requirements': 'NPK ಅವಶ್ಯಕತೆಗಳು',
        'Nitrogen': 'ಸಾರಜನಕ',
        'Phosphorus': 'ರಂಜಕ',
        'Potassium': 'ಪೊಟ್ಯಾಸಿಯಮ್',
        'Recommended Dose': 'ಶಿಫಾರಸು ಮಾಡಲಾದ ಪ್ರಮಾಣ',
        'Approximate Bags Needed (per acre)': 'ಅಂದಾಜು ಚೀಲಗಳು ಬೇಕಾಗುತ್ತವೆ (ಪ್ರತಿ ಎಕರೆಗೆ)',
        'Exact quantity depends on your soil test results': 'ನಿಖರ ಪ್ರಮಾಣವು ನಿಮ್ಮ ಮಣ್ಣಿನ ಪರೀಕ್ಷೆಯ ಫಲಿತಾಂಶಗಳ ಮೇಲೆ ಅವಲಂಬಿತವಾಗಿದೆ',
        'Your Soil Analysis': 'ನಿಮ್ಮ ಮಣ್ಣಿನ ವಿಶ್ಲೇಷಣೆ',
        'Location': 'ಸ್ಥಳ',
        'pH Level': 'pH ಮಟ್ಟ',
        'Moisture': 'ತೇವಾಂಶ',
        'Temperature': 'ತಾಪಮಾನ',
        'Your soil is': 'ನಿಮ್ಮ ಮಣ್ಣು',
        'acidic': 'ಆಮ್ಲೀಯ',
        'alkaline': 'ಕ್ಷಾರೀಯ',
        'Consider adding lime to raise pH': 'pH ಹೆಚ್ಚಿಸಲು ಸುಣ್ಣವನ್ನು ಸೇರಿಸುವುದನ್ನು ಪರಿಗಣಿಸಿ',
        'Consider adding sulfur or organic matter': 'ಗಂಧಕ ಅಥವಾ ಸಾವಯವ ಪದಾರ್ಥವನ್ನು ಸೇರಿಸುವುದನ್ನು ಪರಿಗಣಿಸಿ',
        'Your soil pH is in the optimal range': 'ನಿಮ್ಮ ಮಣ್ಣಿನ pH ಸೂಕ್ತ ವ್ಯಾಪ್ತಿಯಲ್ಲಿದೆ',
        'Growing Regions': 'ಬೆಳೆಯುವ ಪ್ರದೇಶಗಳು',
        'Soil pH': 'ಮಣ್ಣಿನ pH',
        'Economic Value': 'ಆರ್ಥಿಕ ಮೌಲ್ಯ',
        'Your Top Crop Recommendations': 'ನಿಮ್ಮ ಪ್ರಮುಖ ಬೆಳೆ ಶಿಫಾರಸುಗಳು',
        'match': 'ಹೊಂದಾಣಿಕೆ',
        'These are based on your soil parameters and location': 'ಇವು ನಿಮ್ಮ ಮಣ್ಣಿನ ನಿಯತಾಂಕಗಳು ಮತ್ತು ಸ್ಥಳದ ಆಧಾರದ ಮೇಲೆ',
        'Tell me about': 'ಬಗ್ಗೆ ಹೇಳಿ',
        'the top crop': 'ಪ್ರಮುಖ ಬೆಳೆ',
        'Show profitability comparison': 'ಲಾಭದಾಯಕತೆ ತುಲನೆ ತೋರಿಸಿ',
        'What fertilizer do I need': 'ನನಗೆ ಯಾವ ಗೊಬ್ಬರ ಬೇಕು',
        'Namaste! I\'m FarmerHelp': 'ನಮಸ್ಕಾರ! ನಾನು ರೈತ ಸಹಾಯ',
        'I can assist you with': 'ನಾನು ನಿಮಗೆ ಸಹಾಯ ಮಾಡಬಲ್ಲೆ',
        'Weather & seasonal information': 'ಹವಾಮಾನ ಮತ್ತು ಋತುಮಾನ ಮಾಹಿತಿ',
        'Profitability analysis': 'ಲಾಭದಾಯಕತೆ ವಿಶ್ಲೇಷಣೆ',
        'Fertilizer calculations': 'ಗೊಬ್ಬರ ಲೆಕ್ಕಾಚಾರಗಳು',
        'Soil requirements': 'ಮಣ್ಣಿನ ಅವಶ್ಯಕತೆಗಳು',
        'Crop recommendations': 'ಬೆಳೆ ಶಿಫಾರಸುಗಳು',
        'Just ask me anything about farming in Karnataka': 'ಕರ್ನಾಟಕದಲ್ಲಿ ಕೃಷಿಯ ಬಗ್ಗೆ ನನಗೆ ಏನು ಬೇಕಾದರೂ ಕೇಳಿ',
        'Low': 'ಕಡಿಮೆ',
        'Medium': 'ಮಧ್ಯಮ',
        'High': 'ಹೆಚ್ಚು',
        'Rising': 'ಏರುತ್ತಿದೆ',
        'Stable': 'ಸ್ಥಿರ',
        'Falling': 'ಕುಸಿಯುತ್ತಿದೆ',
        'per acre': 'ಪ್ರತಿ ಎಕರೆಗೆ',
        'Risk': 'ಅಪಾಯ',
    }
    
    # Crop name translations
    crop_translations = {
        'coconut': 'ತೆಂಗಿನಕಾಯಿ',
        'cotton': 'ಹತ್ತಿ',
        'coffee': 'ಕಾಫಿ',
        'sugarcane': 'ಕಬ್ಬು',
        'mustard': 'ಸಾಸಿವೆ',
        'soybean': 'ಸೋಯಾಬೀನ್',
        'chillies': 'ಮೆಣಸಿನಕಾಯಿ',
        'cardamom': 'ಏಲಕ್ಕಿ',
        'tea': 'ಚಹಾ',
        'arecanut': 'ಅಡಿಕೆ',
        'mulberry': 'ಹಿಪ್ಪುನೇರಳೆ',
        'ragi': 'ರಾಗಿ',
        'turmeric': 'ಅರಿಶಿನ'
    }
    
    return translations, crop_translations


def generate_farmerhelp_response(message, context, language='en'):
    """Generate contextual responses based on user query"""
    import re
    
    # Extract context data
    user_inputs = context.get('userInputs', {})
    recommendations = context.get('recommendations', [])
    location = context.get('location', 'Karnataka')
    
    # Get translations if Kannada
    if language == 'kn':
        translations, crop_translations = translate_to_kannada({})
    
    # Default suggestions
    if language == 'kn':
        default_suggestions = [
            '🌦️ ಹವಾಮಾನ ಹೇಗಿದೆ?',
            '💰 ಯಾವ ಬೆಳೆ ಹೆಚ್ಚು ಲಾಭದಾಯಕ?',
            '🌿 ಎಷ್ಟು ಗೊಬ್ಬರ ಬೇಕು?',
            '🧪 ಮಣ್ಣಿನ ಅವಶ್ಯಕತೆಗಳ ಬಗ್ಗೆ ಹೇಳಿ'
        ]
    else:
        default_suggestions = [
            '🌦️ What is the weather like?',
            '💰 Which crop is most profitable?',
            '🌿 How much fertilizer needed?',
            '🧪 Tell me about soil requirements'
        ]
    
    # Weather queries
    if any(word in message for word in ['weather', 'climate', 'temperature', 'rain', 'season', 'ಹವಾಮಾನ', 'ಋತು', 'ತಾಪಮಾನ', 'ಮಳೆ']):
        season_info = get_season_info()
        
        if language == 'kn':
            season_names = {'Kharif': 'ಖರೀಫ್', 'Rabi': 'ರಬೀ', 'Summer / Zaid': 'ಬೇಸಿಗೆ / ಝಾಯಿದ್'}
            season_kn = season_names.get(season_info['season'], season_info['season'])
            
            response = f"🌦️ <strong>ಪ್ರಸ್ತುತ ಋತು: {season_kn}</strong><br><br>"
            
            # Translate message
            if season_info['season'] == 'Kharif':
                response += f"ಮಾನ್ಸೂನ್ ಋತು ಸಕ್ರಿಯವಾಗಿದೆ - ಬೆಳೆಗಳನ್ನು ಬಿತ್ತಲು ಸೂಕ್ತ ಸಮಯ.<br><br>"
            elif season_info['season'] == 'Rabi':
                response += f"ರಬೀ ಋತು ಸಕ್ರಿಯವಾಗಿದೆ - ತಂಪಾದ ಹವಾಮಾನ ಮತ್ತು ನೀರಾವರಿ ಬೆಳೆಗಳಿಗೆ ಸೂಕ್ತ.<br><br>"
            else:
                response += f"ಬೇಸಿಗೆ ಋತು - ಸೀಮಿತ ಬೆಳೆಗಳು ಸೂಕ್ತ; ದೀರ್ಘಕಾಲಿಕ ಬೆಳೆಗಳ ನಿರ್ವಹಣೆಗೆ ಗಮನ ನೀಡಿ.<br><br>"
            
            # Translate crop names
            crops_kn = []
            for crop in season_info['recommended_crops']:
                crop_lower = crop.lower()
                if crop_lower in crop_translations:
                    crops_kn.append(crop_translations[crop_lower])
                else:
                    crops_kn.append(crop)
            
            response += f"<strong>ಶಿಫಾರಸು ಮಾಡಲಾದ ಬೆಳೆಗಳು:</strong> {', '.join(crops_kn)}<br><br>"
            response += f"💡 <strong>ಸಲಹೆ:</strong> {season_info['tip']}"
            
            suggestions = [
                f"{crops_kn[0]} ಬಗ್ಗೆ ಹೇಳಿ",
                "ಮಣ್ಣಿನ ಅವಶ್ಯಕತೆಗಳು ಏನು?",
                "ಲಾಭದಾಯಕತೆ ವಿಶ್ಲೇಷಣೆ ತೋರಿಸಿ"
            ]
        else:
            response = f"🌦️ <strong>Current Season: {season_info['season']}</strong><br><br>"
            response += f"{season_info['message']}<br><br>"
            response += f"<strong>Recommended crops:</strong> {', '.join(season_info['recommended_crops'])}<br><br>"
            response += f"💡 <strong>Tip:</strong> {season_info['tip']}"
            
            suggestions = [
                f"Tell me about {season_info['recommended_crops'][0]}",
                "What are the soil requirements?",
                "Show me profitability analysis"
            ]
        return response, suggestions
    
    # Seasonal suitability queries
    if any(word in message.lower() for word in ['season', 'suitable', 'when', 'grow', 'plant', 'sow', 'ಋತು', 'ಸೂಕ್ತ', 'ಯಾವಾಗ', 'ಬೆಳೆಯಿರಿ', 'ಬಿತ್ತಿರಿ']):
        # Extract crop names from message
        mentioned_crops = []
        message_lower = message.lower()
        
        # Check English crop names
        for crop in CROP_INFO.keys():
            if crop in message_lower or crop.replace('_', ' ') in message_lower:
                mentioned_crops.append(crop)
        
        # Check Kannada crop names if in Kannada mode
        if language == 'kn':
            for crop_key, crop_kn in crop_translations.items():
                if crop_kn in message:
                    if crop_key not in mentioned_crops:
                        mentioned_crops.append(crop_key)
        
        if mentioned_crops:
            # Build seasonal information for mentioned crops
            crop_seasons = {}
            for crop in mentioned_crops:
                if crop in PROFIT_DATA:
                    season = PROFIT_DATA[crop]['best_season']
                    crop_seasons[crop] = season
            
            if crop_seasons:
                if language == 'kn':
                    season_names = {'Kharif': 'ಖರೀಫ್ (ಮಾನ್ಸೂನ್)', 'Rabi': 'ರಬೀ (ಚಳಿಗಾಲ)', 'Both': 'ಎರಡೂ ಋತುಗಳು', 'Summer / Zaid': 'ಬೇಸಿಗೆ'}
                    
                    if len(mentioned_crops) == 1:
                        crop = mentioned_crops[0]
                        crop_kn = crop_translations.get(crop, crop.capitalize())
                        season = crop_seasons[crop]
                        season_kn = season_names.get(season, season)
                        
                        response = f"🌱 <strong>{crop_kn} ಗಾಗಿ ಸೂಕ್ತ ಋತು</strong><br><br>"
                        response += f"<strong>ಉತ್ತಮ ಋತು:</strong> {season_kn}<br><br>"
                        
                        if season == 'Kharif':
                            response += "📅 <strong>ಬಿತ್ತನೆ ಸಮಯ:</strong> ಜೂನ್-ಜುಲೈ (ಮಾನ್ಸೂನ್ ಆರಂಭ)<br>"
                            response += "🌾 <strong>ಕೊಯ್ಲು ಸಮಯ:</strong> ಸೆಪ್ಟೆಂಬರ್-ಅಕ್ಟೋಬರ್<br><br>"
                            response += "💡 ಖರೀಫ್ ಬೆಳೆಗಳು ಹೆಚ್ಚಿನ ತಾಪಮಾನ ಮತ್ತು ಮಾನ್ಸೂನ್ ಮಳೆಯ ಮೇಲೆ ಅವಲಂಬಿತವಾಗಿವೆ. ಆರಂಭಿಕ ಬೆಳವಣಿಗೆ ಹಂತದಲ್ಲಿ ಸ್ಥಿರ ಮಳೆ ಅಗತ್ಯ."
                        elif season == 'Rabi':
                            response += "📅 <strong>ಬಿತ್ತನೆ ಸಮಯ:</strong> ಅಕ್ಟೋಬರ್-ನವೆಂಬರ್ (ಮಾನ್ಸೂನ್ ನಂತರ)<br>"
                            response += "🌾 <strong>ಕೊಯ್ಲು ಸಮಯ:</strong> ಮಾರ್ಚ್-ಏಪ್ರಿಲ್<br><br>"
                            response += "💡 ರಬೀ ಬೆಳೆಗಳು ತಂಪಾದ, ಶುಷ್ಕ ಹವಾಮಾನದಲ್ಲಿ ಬೆಳೆಯುತ್ತವೆ. ಮಾನ್ಸೂನ್ ನಂತರದ ಮಣ್ಣಿನ ತೇವಾಂಶ ಅಥವಾ ನೀರಾವರಿಯ ಮೇಲೆ ಅವಲಂಬಿತವಾಗಿವೆ."
                        else:  # Both or perennial
                            response += "📅 <strong>ಬಿತ್ತನೆ ಸಮಯ:</strong> ವರ್ಷಪೂರ್ತಿ (ನೀರಾವರಿ ಲಭ್ಯವಿದ್ದರೆ)<br><br>"
                            if crop in ['coconut', 'arecanut', 'cardamom', 'coffee', 'tea']:
                                response += "💡 ಇದು ದೀರ್ಘಕಾಲಿಕ ತೋಟಗಾರಿಕೆ ಬೆಳೆ. ಮಾನ್ಸೂನ್ ಮಳೆಯು ಅಗತ್ಯವಾಗಿದ್ದರೂ, ಕೊಯ್ಲು ವರ್ಷಪೂರ್ತಿ ಸಂಭವಿಸಬಹುದು. ಹಲವು ವರ್ಷಗಳವರೆಗೆ ಉತ್ಪಾದಕವಾಗಿರುತ್ತದೆ."
                            elif crop == 'sugarcane':
                                response += "💡 ಕಬ್ಬು ವಾರ್ಷಿಕ ಬೆಳೆಯಾಗಿದ್ದು, ಸಾಮಾನ್ಯವಾಗಿ ಬೇಸಿಗೆ ಅಥವಾ ಮಾನ್ಸೂನ್‌ನಲ್ಲಿ ನೆಡಲಾಗುತ್ತದೆ ಮತ್ತು ಮುಂದಿನ ಚಳಿಗಾಲದಲ್ಲಿ ಕೊಯ್ಲು ಮಾಡಲಾಗುತ್ತದೆ. ಸಂಪೂರ್ಣ ವರ್ಷ ಹೊಲದಲ್ಲಿರುತ್ತದೆ."
                            elif crop == 'mulberry':
                                response += "💡 ಹಿಪ್ಪುನೇರಳೆ ದೀರ್ಘಕಾಲಿಕ ಬೆಳೆಯಾಗಿದ್ದು, ಬೇಸಿಗೆ ಮತ್ತು ಚಳಿಗಾಲ ಸೇರಿದಂತೆ ಹಲವು ಋತುಗಳಲ್ಲಿ ಎಲೆ ಕೊಯ್ಲು ಮಾಡಬಹುದು."
                            else:
                                response += "💡 ಈ ಬೆಳೆಯನ್ನು ಎರಡೂ ಋತುಗಳಲ್ಲಿ ಬೆಳೆಯಬಹುದು, ಆದರೆ ಸಾಕಷ್ಟು ನೀರು ಅಗತ್ಯ."
                        
                        suggestions = [
                            f"{crop_kn} ಗೆ ಗೊಬ್ಬರ ಅವಶ್ಯಕತೆಗಳು",
                            f"{crop_kn} ಲಾಭದಾಯಕತೆ",
                            "ಇತರ ಬೆಳೆಗಳ ಋತುಗಳು"
                        ]
                    else:
                        # Multiple crops
                        response = f"🌱 <strong>ಬೆಳೆಗಳ ಸೂಕ್ತ ಋತುಗಳು</strong><br><br>"
                        
                        for crop in mentioned_crops:
                            crop_kn = crop_translations.get(crop, crop.capitalize())
                            season = crop_seasons[crop]
                            season_kn = season_names.get(season, season)
                            response += f"• <strong>{crop_kn}:</strong> {season_kn}<br>"
                        
                        response += "<br>💡 ಪ್ರತಿ ಬೆಳೆಗೆ ವಿವರವಾದ ಮಾಹಿತಿಗಾಗಿ ಪ್ರತ್ಯೇಕವಾಗಿ ಕೇಳಿ."
                        
                        suggestions = [
                            f"{crop_translations.get(mentioned_crops[0], mentioned_crops[0].capitalize())} ಬಗ್ಗೆ ಹೇಳಿ",
                            "ಪ್ರಸ್ತುತ ಋತು ಏನು?",
                            "ಲಾಭದಾಯಕ ಬೆಳೆಗಳು"
                        ]
                else:
                    # English response
                    season_names = {'Kharif': 'Kharif (Monsoon)', 'Rabi': 'Rabi (Winter)', 'Both': 'Both Seasons', 'Summer / Zaid': 'Summer'}
                    
                    if len(mentioned_crops) == 1:
                        crop = mentioned_crops[0]
                        season = crop_seasons[crop]
                        season_name = season_names.get(season, season)
                        
                        response = f"🌱 <strong>Best Season for {crop.capitalize()}</strong><br><br>"
                        response += f"<strong>Optimal Season:</strong> {season_name}<br><br>"
                        
                        if season == 'Kharif':
                            response += "📅 <strong>Sowing Time:</strong> June-July (Monsoon onset)<br>"
                            response += "🌾 <strong>Harvest Time:</strong> September-October<br><br>"
                            response += "💡 Kharif crops require high temperatures and steady monsoon rainfall during early growth stages. They are heavily dependent on the Southwest monsoon for successful yields."
                        elif season == 'Rabi':
                            response += "📅 <strong>Sowing Time:</strong> October-November (Post-monsoon)<br>"
                            response += "🌾 <strong>Harvest Time:</strong> March-April<br><br>"
                            response += "💡 Rabi crops thrive in cool, dry winter climate. They rely on residual soil moisture or controlled irrigation rather than direct rainfall."
                        else:  # Both or perennial
                            response += "📅 <strong>Sowing Time:</strong> Year-round (with irrigation)<br><br>"
                            if crop in ['coconut', 'arecanut', 'cardamom', 'coffee', 'tea']:
                                response += "💡 This is a perennial plantation crop that remains productive for many years. While associated with monsoon due to humidity requirements, harvests can occur throughout the year."
                            elif crop == 'sugarcane':
                                response += "💡 Sugarcane is an annual crop typically planted in summer or late monsoon and harvested during the following winter. It occupies the field for an entire year."
                            elif crop == 'mulberry':
                                response += "💡 Mulberry is a hardy perennial that can provide leaf harvests across multiple seasons, including summer and winter, depending on local irrigation."
                            else:
                                response += "💡 This crop can be grown in both seasons with adequate water supply."
                        
                        suggestions = [
                            f"Fertilizer requirements for {crop}",
                            f"Profitability of {crop}",
                            "Other crop seasons"
                        ]
                    else:
                        # Multiple crops
                        response = f"🌱 <strong>Suitable Seasons for Crops</strong><br><br>"
                        
                        for crop in mentioned_crops:
                            season = crop_seasons[crop]
                            season_name = season_names.get(season, season)
                            response += f"• <strong>{crop.capitalize()}:</strong> {season_name}<br>"
                        
                        response += "<br>💡 Ask about each crop individually for detailed information."
                        
                        suggestions = [
                            f"Tell me about {mentioned_crops[0]}",
                            "What is the current season?",
                            "Most profitable crops"
                        ]
                
                return response, suggestions
    
    # Profitability queries
    if any(word in message for word in ['profit', 'money', 'income', 'revenue', 'earn', 'cost', 'ಲಾಭ', 'ಹಣ', 'ಆದಾಯ', 'ಬೆಲೆ']):
        # Extract all crop names from message
        crops_mentioned = []
        for crop in CROP_INFO.keys():
            if crop in message or crop.replace('_', ' ') in message:
                crops_mentioned.append(crop)
        
        # Check Kannada crop names
        if language == 'kn':
            for crop_key, crop_kn in crop_translations.items():
                if crop_kn in message and crop_key not in crops_mentioned:
                    crops_mentioned.append(crop_key)
        
        # If no crop mentioned, use recommended crop
        if not crops_mentioned and recommendations and len(recommendations) > 0:
            crops_mentioned = [recommendations[0].get('crop', '').lower()]
        
        if crops_mentioned and len(crops_mentioned) > 0:
            # Check if all mentioned crops are valid
            valid_crops = [c for c in crops_mentioned if c in PROFIT_DATA]
            
            if len(valid_crops) == 1:
                # Single crop response
                crop_name = valid_crops[0]
                pdata = PROFIT_DATA[crop_name]
                
                if language == 'kn':
                    crop_kn = crop_translations.get(crop_name, crop_name.capitalize())
                    response = f"💰 <strong>{crop_kn} ಗಾಗಿ ಲಾಭದಾಯಕತೆ ವಿಶ್ಲೇಷಣೆ</strong><br><br>"
                    response += f"<strong>ಪ್ರತಿ ಎಕರೆಗೆ ನಿರೀಕ್ಷಿತ ಲಾಭ:</strong><br>"
                    response += f"• ಕನಿಷ್ಠ: ₹{pdata['min_revenue'] - pdata['input_cost']:,}<br>"
                    response += f"• ಗರಿಷ್ಠ: ₹{pdata['max_revenue'] - pdata['input_cost']:,}<br><br>"
                    response += f"<strong>ಇನ್‌ಪುಟ್ ವೆಚ್ಚ:</strong> ₹{pdata['input_cost']:,}/ಎಕರೆ<br>"
                    
                    risk_kn = {'Low': 'ಕಡಿಮೆ', 'Medium': 'ಮಧ್ಯಮ', 'High': 'ಹೆಚ್ಚು'}.get(pdata['risk_level'], pdata['risk_level'])
                    trend_kn = {'Rising': 'ಏರುತ್ತಿದೆ', 'Stable': 'ಸ್ಥಿರ', 'Falling': 'ಕುಸಿಯುತ್ತಿದೆ'}.get(pdata['price_trend'], pdata['price_trend'])
                    
                    response += f"<strong>ಅಪಾಯ ಮಟ್ಟ:</strong> {risk_kn}<br>"
                    response += f"<strong>ಬೆಲೆ ಪ್ರವೃತ್ತಿ:</strong> {trend_kn}<br><br>"
                    response += f"💡 <strong>ಪ್ರಮುಖ ಸಲಹೆ:</strong> {pdata['key_tips'][0]}"
                    
                    suggestions = [
                        f"{crop_kn} ಗೆ ಎಷ್ಟು ಗೊಬ್ಬರ?",
                        f"{crop_kn} ಮಣ್ಣಿನ ಅವಶ್ಯಕತೆಗಳು ಏನು?",
                        "ಇತರ ಬೆಳೆಗಳೊಂದಿಗೆ ಹೋಲಿಸಿ"
                    ]
                else:
                    response = f"💰 <strong>Profitability Analysis for {crop_name.capitalize()}</strong><br><br>"
                    response += f"<strong>Expected Profit per Acre:</strong><br>"
                    response += f"• Minimum: ₹{pdata['min_revenue'] - pdata['input_cost']:,}<br>"
                    response += f"• Maximum: ₹{pdata['max_revenue'] - pdata['input_cost']:,}<br><br>"
                    response += f"<strong>Input Cost:</strong> ₹{pdata['input_cost']:,}/acre<br>"
                    response += f"<strong>Risk Level:</strong> {pdata['risk_level']}<br>"
                    response += f"<strong>Price Trend:</strong> {pdata['price_trend']}<br><br>"
                    response += f"💡 <strong>Top Tip:</strong> {pdata['key_tips'][0]}"
                    
                    suggestions = [
                        f"How much fertilizer for {crop_name}?",
                        f"What are {crop_name} soil requirements?",
                        "Compare with other crops"
                    ]
                return response, suggestions
            
            elif len(valid_crops) > 1:
                # Multiple crops response
                if language == 'kn':
                    response = f"💰 <strong>ಬೆಳೆಗಳ ಲಾಭದಾಯಕತೆ ತುಲನೆ</strong><br><br>"
                    
                    for crop in valid_crops:
                        pdata = PROFIT_DATA[crop]
                        crop_kn = crop_translations.get(crop, crop.capitalize())
                        avg_profit = (pdata['min_revenue'] + pdata['max_revenue']) // 2 - pdata['input_cost']
                        risk_kn = {'Low': 'ಕಡಿಮೆ', 'Medium': 'ಮಧ್ಯಮ', 'High': 'ಹೆಚ್ಚು'}.get(pdata['risk_level'], pdata['risk_level'])
                        
                        response += f"<strong>{crop_kn}:</strong><br>"
                        response += f"• ಸರಾಸರಿ ಲಾಭ: ₹{avg_profit:,}/ಎಕರೆ<br>"
                        response += f"• ಅಪಾಯ: {risk_kn}<br>"
                        response += f"• ವೆಚ್ಚ: ₹{pdata['input_cost']:,}/ಎಕರೆ<br><br>"
                    
                    response += "💡 <em>ಪ್ರತಿ ಬೆಳೆಗೆ ವಿವರವಾದ ವಿಶ್ಲೇಷಣೆಗಾಗಿ ಪ್ರತ್ಯೇಕವಾಗಿ ಕೇಳಿ.</em>"
                    
                    suggestions = [
                        f"{crop_translations.get(valid_crops[0], valid_crops[0].capitalize())} ಬಗ್ಗೆ ಹೇಳಿ",
                        "ಗೊಬ್ಬರ ಅವಶ್ಯಕತೆಗಳು",
                        "ಮಣ್ಣಿನ ಅವಶ್ಯಕತೆಗಳು"
                    ]
                else:
                    response = f"💰 <strong>Profitability Comparison for Multiple Crops</strong><br><br>"
                    
                    for crop in valid_crops:
                        pdata = PROFIT_DATA[crop]
                        avg_profit = (pdata['min_revenue'] + pdata['max_revenue']) // 2 - pdata['input_cost']
                        
                        response += f"<strong>{crop.capitalize()}:</strong><br>"
                        response += f"• Average Profit: ₹{avg_profit:,}/acre<br>"
                        response += f"• Risk: {pdata['risk_level']}<br>"
                        response += f"• Cost: ₹{pdata['input_cost']:,}/acre<br><br>"
                    
                    response += "💡 <em>Ask about each crop individually for detailed analysis.</em>"
                    
                    suggestions = [
                        f"Tell me about {valid_crops[0]}",
                        "Fertilizer requirements",
                        "Soil requirements"
                    ]
                return response, suggestions
        
        # General profitability
        top_profitable = ['Arecanut', 'Turmeric', 'Chillies', 'Coffee', 'Cardamom']
        
        if language == 'kn':
            response = "💰 <strong>ಕರ್ನಾಟಕದಲ್ಲಿ ಅತ್ಯಂತ ಲಾಭದಾಯಕ ಬೆಳೆಗಳು:</strong><br><br>"
            for i, crop in enumerate(top_profitable[:3], 1):
                crop_key = crop.lower()
                if crop_key in PROFIT_DATA:
                    pdata = PROFIT_DATA[crop_key]
                    avg_profit = (pdata['min_revenue'] + pdata['max_revenue']) // 2 - pdata['input_cost']
                    crop_kn = crop_translations.get(crop_key, crop)
                    risk_kn = {'Low': 'ಕಡಿಮೆ', 'Medium': 'ಮಧ್ಯಮ', 'High': 'ಹೆಚ್ಚು'}.get(pdata['risk_level'], pdata['risk_level'])
                    response += f"{i}. <strong>{crop_kn}</strong> - ₹{avg_profit:,}/ಎಕರೆ (ಅಪಾಯ: {risk_kn})<br>"
            
            response += "<br>💡 ಲಾಭದಾಯಕತೆಯು ನಿಮ್ಮ ಮಣ್ಣು, ಹವಾಮಾನ ಮತ್ತು ಮಾರುಕಟ್ಟೆ ಪ್ರವೇಶವನ್ನು ಅವಲಂಬಿಸಿರುತ್ತದೆ!"
            
            suggestions = [
                "ಅಡಿಕೆ ಬಗ್ಗೆ ಹೇಳಿ",
                "ಮಣ್ಣಿನ ಅವಶ್ಯಕತೆಗಳು ಏನು?",
                "ಹವಾಮಾನ ಮಾಹಿತಿ ತೋರಿಸಿ"
            ]
        else:
            response = "💰 <strong>Most Profitable Crops in Karnataka:</strong><br><br>"
            for i, crop in enumerate(top_profitable[:3], 1):
                crop_key = crop.lower()
                if crop_key in PROFIT_DATA:
                    pdata = PROFIT_DATA[crop_key]
                    avg_profit = (pdata['min_revenue'] + pdata['max_revenue']) // 2 - pdata['input_cost']
                    response += f"{i}. <strong>{crop}</strong> - ₹{avg_profit:,}/acre (Risk: {pdata['risk_level']})<br>"
            
            response += "<br>💡 Profitability depends on your soil, climate, and market access!"
            
            suggestions = [
                "Tell me about Arecanut",
                "What are soil requirements?",
                "Show weather information"
            ]
        
        return response, suggestions
    
    # Fertilizer queries
    if any(word in message for word in ['fertilizer', 'fertiliser', 'npk', 'nitrogen', 'phosphorus', 'potassium', 'bags', 'ಗೊಬ್ಬರ', 'ಸಾರಜನಕ', 'ರಂಜಕ', 'ಪೊಟ್ಯಾಸಿಯಮ್']):
        # Extract all crop names from message
        crops_mentioned = []
        for crop in CROP_INFO.keys():
            if crop in message or crop.replace('_', ' ') in message:
                crops_mentioned.append(crop)
        
        # Check Kannada crop names
        if language == 'kn':
            for crop_key, crop_kn in crop_translations.items():
                if crop_kn in message and crop_key not in crops_mentioned:
                    crops_mentioned.append(crop_key)
        
        # If no crop mentioned, use recommended crop
        if not crops_mentioned and recommendations and len(recommendations) > 0:
            crops_mentioned = [recommendations[0].get('crop', '').lower()]
        
        if crops_mentioned and len(crops_mentioned) > 0:
            # Check if all mentioned crops are valid
            valid_crops = [c for c in crops_mentioned if c in CROP_INFO]
            
            if len(valid_crops) == 1:
                # Single crop response
                crop_mentioned = valid_crops[0]
                info = CROP_INFO[crop_mentioned]
                
                if language == 'kn':
                    crop_kn = crop_translations.get(crop_mentioned, crop_mentioned.capitalize())
                    response = f"🌿 <strong>{crop_kn} ಗಾಗಿ ಗೊಬ್ಬರ ಅವಶ್ಯಕತೆಗಳು</strong><br><br>"
                    response += f"<strong>NPK ಅವಶ್ಯಕತೆಗಳು:</strong><br>"
                    response += f"• ಸಾರಜನಕ (N): {info['npk_ranges']['N']}<br>"
                    response += f"• ರಂಜಕ (P): {info['npk_ranges']['P']}<br>"
                    response += f"• ಪೊಟ್ಯಾಸಿಯಮ್ (K): {info['npk_ranges']['K']}<br><br>"
                    response += f"<strong>ಶಿಫಾರಸು ಮಾಡಲಾದ ಪ್ರಮಾಣ:</strong><br>{info['fertilizer']}<br><br>"
                    
                    # Calculate bags (approximate)
                    response += "📦 <strong>ಅಂದಾಜು ಚೀಲಗಳು ಬೇಕಾಗುತ್ತವೆ (ಪ್ರತಿ ಎಕರೆಗೆ):</strong><br>"
                    response += "• ಯೂರಿಯಾ (46% N): 2-4 ಚೀಲಗಳು (ಪ್ರತಿಯೊಂದೂ 50kg)<br>"
                    response += "• DAP (18% N, 46% P): 1-2 ಚೀಲಗಳು<br>"
                    response += "• MOP (60% K): 1-2 ಚೀಲಗಳು<br><br>"
                    response += "💡 <em>ನಿಖರ ಪ್ರಮಾಣವು ನಿಮ್ಮ ಮಣ್ಣಿನ ಪರೀಕ್ಷೆಯ ಫಲಿತಾಂಶಗಳ ಮೇಲೆ ಅವಲಂಬಿತವಾಗಿದೆ!</em>"
                    
                    suggestions = [
                        f"{crop_kn} ಗೆ ಲಾಭ ಎಷ್ಟು?",
                        "ಗೊಬ್ಬರ ಯಾವಾಗ ಹಾಕಬೇಕು?",
                        "ಮಣ್ಣಿನ pH ಬಗ್ಗೆ ಹೇಳಿ"
                    ]
                else:
                    response = f"🌿 <strong>Fertilizer Requirements for {crop_mentioned.capitalize()}</strong><br><br>"
                    response += f"<strong>NPK Requirements:</strong><br>"
                    response += f"• Nitrogen (N): {info['npk_ranges']['N']}<br>"
                    response += f"• Phosphorus (P): {info['npk_ranges']['P']}<br>"
                    response += f"• Potassium (K): {info['npk_ranges']['K']}<br><br>"
                    response += f"<strong>Recommended Dose:</strong><br>{info['fertilizer']}<br><br>"
                    
                    # Calculate bags (approximate)
                    response += "📦 <strong>Approximate Bags Needed (per acre):</strong><br>"
                    response += "• Urea (46% N): 2-4 bags (50kg each)<br>"
                    response += "• DAP (18% N, 46% P): 1-2 bags<br>"
                    response += "• MOP (60% K): 1-2 bags<br><br>"
                    response += "💡 <em>Exact quantity depends on your soil test results!</em>"
                    
                    suggestions = [
                        f"What is the profit for {crop_mentioned}?",
                        "When should I apply fertilizer?",
                        "Tell me about soil pH"
                    ]
                return response, suggestions
            
            elif len(valid_crops) > 1:
                # Multiple crops response
                if language == 'kn':
                    response = f"🌿 <strong>ಬೆಳೆಗಳ ಗೊಬ್ಬರ ಅವಶ್ಯಕತೆಗಳು</strong><br><br>"
                    
                    for crop in valid_crops:
                        info = CROP_INFO[crop]
                        crop_kn = crop_translations.get(crop, crop.capitalize())
                        response += f"<strong>{crop_kn}:</strong><br>"
                        response += f"• N: {info['npk_ranges']['N']}, P: {info['npk_ranges']['P']}, K: {info['npk_ranges']['K']}<br>"
                        response += f"• ಪ್ರಮಾಣ: {info['fertilizer']}<br><br>"
                    
                    response += "💡 <em>ಪ್ರತಿ ಬೆಳೆಗೆ ವಿವರವಾದ ಮಾಹಿತಿಗಾಗಿ ಪ್ರತ್ಯೇಕವಾಗಿ ಕೇಳಿ.</em>"
                    
                    suggestions = [
                        f"{crop_translations.get(valid_crops[0], valid_crops[0].capitalize())} ಬಗ್ಗೆ ಹೇಳಿ",
                        "ಲಾಭದಾಯಕತೆ ತುಲನೆ",
                        "ಮಣ್ಣಿನ ಅವಶ್ಯಕತೆಗಳು"
                    ]
                else:
                    response = f"🌿 <strong>Fertilizer Requirements for Multiple Crops</strong><br><br>"
                    
                    for crop in valid_crops:
                        info = CROP_INFO[crop]
                        response += f"<strong>{crop.capitalize()}:</strong><br>"
                        response += f"• N: {info['npk_ranges']['N']}, P: {info['npk_ranges']['P']}, K: {info['npk_ranges']['K']}<br>"
                        response += f"• Dose: {info['fertilizer']}<br><br>"
                    
                    response += "💡 <em>Ask about each crop individually for detailed information.</em>"
                    
                    suggestions = [
                        f"Tell me about {valid_crops[0]}",
                        "Compare profitability",
                        "Soil requirements"
                    ]
                return response, suggestions
        
        if language == 'kn':
            response = "🌿 <strong>ಗೊಬ್ಬರ ಮಾಹಿತಿ</strong><br><br>"
            response += "ನಿಮಗೆ ನಿಖರವಾದ ಗೊಬ್ಬರ ಶಿಫಾರಸುಗಳನ್ನು ನೀಡಲು, ದಯವಿಟ್ಟು ನನಗೆ ತಿಳಿಸಿ:<br>"
            response += "• ನೀವು ಯಾವ ಬೆಳೆ ಬೆಳೆಯುತ್ತಿದ್ದೀರಿ?<br>"
            response += "• ನಿಮ್ಮ ಭೂಮಿಯ ವಿಸ್ತೀರ್ಣ ಎಷ್ಟು?<br><br>"
            response += "ಅಥವಾ ಕೇಳಿ: '[ಬೆಳೆ ಹೆಸರು] ಗೆ ಎಷ್ಟು ಗೊಬ್ಬರ ಬೇಕು?'"
            
            suggestions = [
                "ರಾಗಿಗೆ ಗೊಬ್ಬರ",
                "ಹತ್ತಿಗೆ ಗೊಬ್ಬರ",
                "ಮಣ್ಣಿನ ಅವಶ್ಯಕತೆಗಳು ತೋರಿಸಿ"
            ]
        else:
            response = "🌿 <strong>Fertilizer Information</strong><br><br>"
            response += "To give you accurate fertilizer recommendations, please tell me:<br>"
            response += "• Which crop are you growing?<br>"
            response += "• What is your land area?<br><br>"
            response += "Or ask: 'How much fertilizer for [crop name]?'"
            
            suggestions = [
                "Fertilizer for Ragi",
                "Fertilizer for Cotton",
                "Show me soil requirements"
            ]
        return response, suggestions
    
    # Soil queries
    if any(word in message for word in ['soil', 'ph', 'moisture', 'nutrient', 'test', 'ಮಣ್ಣು', 'ತೇವಾಂಶ', 'ಪೋಷಕಾಂಶ', 'ಪರೀಕ್ಷೆ']):
        if user_inputs:
            if language == 'kn':
                response = f"🧪 <strong>ನಿಮ್ಮ ಮಣ್ಣಿನ ವಿಶ್ಲೇಷಣೆ</strong><br><br>"
                response += f"<strong>ಸ್ಥಳ:</strong> {location}<br>"
                response += f"<strong>ಸಾರಜನಕ (N):</strong> {user_inputs.get('N', 'N/A')} kg/ha<br>"
                response += f"<strong>ರಂಜಕ (P):</strong> {user_inputs.get('P', 'N/A')} kg/ha<br>"
                response += f"<strong>ಪೊಟ್ಯಾಸಿಯಮ್ (K):</strong> {user_inputs.get('K', 'N/A')} kg/ha<br>"
                response += f"<strong>pH ಮಟ್ಟ:</strong> {user_inputs.get('ph', 'N/A')}<br>"
                response += f"<strong>ತೇವಾಂಶ:</strong> {user_inputs.get('moisture', 'N/A')}%<br>"
                response += f"<strong>ತಾಪಮಾನ:</strong> {user_inputs.get('temp', 'N/A')}°C<br><br>"
                
                # Soil health assessment
                ph = float(user_inputs.get('ph', 7))
                if ph < 5.5:
                    response += "⚠️ ನಿಮ್ಮ ಮಣ್ಣು <strong>ಆಮ್ಲೀಯ</strong>ವಾಗಿದೆ. pH ಹೆಚ್ಚಿಸಲು ಸುಣ್ಣವನ್ನು ಸೇರಿಸುವುದನ್ನು ಪರಿಗಣಿಸಿ.<br>"
                elif ph > 8.0:
                    response += "⚠️ ನಿಮ್ಮ ಮಣ್ಣು <strong>ಕ್ಷಾರೀಯ</strong>ವಾಗಿದೆ. ಗಂಧಕ ಅಥವಾ ಸಾವಯವ ಪದಾರ್ಥವನ್ನು ಸೇರಿಸುವುದನ್ನು ಪರಿಗಣಿಸಿ.<br>"
                else:
                    response += "✅ ನಿಮ್ಮ ಮಣ್ಣಿನ pH <strong>ಸೂಕ್ತ ವ್ಯಾಪ್ತಿಯಲ್ಲಿದೆ</strong>!<br>"
                
                suggestions = [
                    "ನನ್ನ ಮಣ್ಣಿಗೆ ಯಾವ ಬೆಳೆಗಳು ಸೂಕ್ತ?",
                    "ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಹೇಗೆ ಸುಧಾರಿಸುವುದು?",
                    "ಲಾಭದಾಯಕತೆ ವಿಶ್ಲೇಷಣೆ ತೋರಿಸಿ"
                ]
            else:
                response = f"🧪 <strong>Your Soil Analysis</strong><br><br>"
                response += f"<strong>Location:</strong> {location}<br>"
                response += f"<strong>Nitrogen (N):</strong> {user_inputs.get('N', 'N/A')} kg/ha<br>"
                response += f"<strong>Phosphorus (P):</strong> {user_inputs.get('P', 'N/A')} kg/ha<br>"
                response += f"<strong>Potassium (K):</strong> {user_inputs.get('K', 'N/A')} kg/ha<br>"
                response += f"<strong>pH Level:</strong> {user_inputs.get('ph', 'N/A')}<br>"
                response += f"<strong>Moisture:</strong> {user_inputs.get('moisture', 'N/A')}%<br>"
                response += f"<strong>Temperature:</strong> {user_inputs.get('temp', 'N/A')}°C<br><br>"
                
                # Soil health assessment
                ph = float(user_inputs.get('ph', 7))
                if ph < 5.5:
                    response += "⚠️ Your soil is <strong>acidic</strong>. Consider adding lime to raise pH.<br>"
                elif ph > 8.0:
                    response += "⚠️ Your soil is <strong>alkaline</strong>. Consider adding sulfur or organic matter.<br>"
                else:
                    response += "✅ Your soil pH is in the <strong>optimal range</strong>!<br>"
                
                suggestions = [
                    "Which crops suit my soil?",
                    "How to improve soil health?",
                    "Show profitability analysis"
                ]
            return response, suggestions
        
        if language == 'kn':
            response = "🧪 <strong>ಮಣ್ಣಿನ ಪರೀಕ್ಷೆಯ ಮಾಹಿತಿ</strong><br><br>"
            response += "ಸೂಕ್ತ ಬೆಳೆ ಆಯ್ಕೆಗೆ ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ ಅತ್ಯಗತ್ಯ! ಪರೀಕ್ಷಿಸಬೇಕಾದವು:<br>"
            response += "• NPK ಮಟ್ಟಗಳು (ಸಾರಜನಕ, ರಂಜಕ, ಪೊಟ್ಯಾಸಿಯಮ್)<br>"
            response += "• pH ಮಟ್ಟ (ಆಮ್ಲೀಯತೆ/ಕ್ಷಾರೀಯತೆ)<br>"
            response += "• ಸಾವಯವ ಪದಾರ್ಥದ ಅಂಶ<br>"
            response += "• ಸೂಕ್ಷ್ಮ ಪೋಷಕಾಂಶಗಳು (ಸತು, ಕಬ್ಬಿಣ, ಇತ್ಯಾದಿ)<br><br>"
            response += "💡 ನಿಮ್ಮ ಹತ್ತಿರದ ಮಣ್ಣಿನ ಪರೀಕ್ಷಾ ಪ್ರಯೋಗಾಲಯಕ್ಕೆ ಭೇಟಿ ನೀಡಿ. ವೆಚ್ಚ: ಪ್ರತಿ ಮಾದರಿಗೆ ₹5-50."
            
            suggestions = [
                "ಮಣ್ಣಿನ ಪರೀಕ್ಷಾ ಪ್ರಯೋಗಾಲಯಗಳನ್ನು ಹುಡುಕಿ",
                "ಸೂಕ್ತ pH ಎಷ್ಟು?",
                "NPK ಬಗ್ಗೆ ಹೇಳಿ"
            ]
        else:
            response = "🧪 <strong>Soil Testing Information</strong><br><br>"
            response += "Soil testing is crucial for optimal crop selection! Test for:<br>"
            response += "• NPK levels (Nitrogen, Phosphorus, Potassium)<br>"
            response += "• pH level (acidity/alkalinity)<br>"
            response += "• Organic matter content<br>"
            response += "• Micronutrients (Zinc, Iron, etc.)<br><br>"
            response += "💡 Visit your nearest soil testing lab. Cost: ₹5-50 per sample."
            
            suggestions = [
                "Find soil testing labs",
                "What is optimal pH?",
                "Tell me about NPK"
            ]
        return response, suggestions
    
    # Crop-specific queries
    crops_found = []
    for crop in CROP_INFO.keys():
        if crop in message or crop.replace('_', ' ') in message:
            crops_found.append(crop)
    
    # Check Kannada crop names
    if language == 'kn':
        for crop_key, crop_kn in crop_translations.items():
            if crop_kn in message and crop_key not in crops_found:
                crops_found.append(crop_key)
    
    if len(crops_found) == 1:
        # Single crop response
        crop = crops_found[0]
        info = CROP_INFO[crop]
        
        if language == 'kn':
            crop_kn = crop_translations.get(crop, crop.capitalize())
            response = f"🌾 <strong>{crop_kn}</strong><br>"
            response += f"<em>{info['kannada_name']}</em><br><br>"
            response += f"{info['desc']}<br><br>"
            response += f"<strong>ಬೆಳೆಯುವ ಪ್ರದೇಶಗಳು:</strong> {info['regions']}<br>"
            response += f"<strong>ತಾಪಮಾನ:</strong> {info['temp_range']}<br>"
            response += f"<strong>ಮಣ್ಣಿನ pH:</strong> {info['ph_range']}<br>"
            response += f"<strong>ತೇವಾಂಶ:</strong> {info['moisture_range']}<br><br>"
            response += f"💰 <strong>ಆರ್ಥಿಕ ಮೌಲ್ಯ:</strong> {info['profit']}"
            
            suggestions = [
                f"{crop_kn} ಗೆ ಗೊಬ್ಬರ",
                f"{crop_kn} ಗೆ ಲಾಭ ವಿಶ್ಲೇಷಣೆ",
                "ಇತರ ಬೆಳೆಗಳೊಂದಿಗೆ ಹೋಲಿಸಿ"
            ]
        else:
            response = f"🌾 <strong>{crop.capitalize()}</strong><br>"
            response += f"<em>{info['kannada_name']}</em><br><br>"
            response += f"{info['desc']}<br><br>"
            response += f"<strong>Growing Regions:</strong> {info['regions']}<br>"
            response += f"<strong>Temperature:</strong> {info['temp_range']}<br>"
            response += f"<strong>Soil pH:</strong> {info['ph_range']}<br>"
            response += f"<strong>Moisture:</strong> {info['moisture_range']}<br><br>"
            response += f"💰 <strong>Economic Value:</strong> {info['profit']}"
            
            suggestions = [
                f"Fertilizer for {crop}",
                f"Profit analysis for {crop}",
                "Compare with other crops"
            ]
        return response, suggestions
    
    elif len(crops_found) > 1:
        # Multiple crops response
        if language == 'kn':
            response = f"🌾 <strong>ಬೆಳೆಗಳ ಮಾಹಿತಿ</strong><br><br>"
            
            for crop in crops_found:
                info = CROP_INFO[crop]
                crop_kn = crop_translations.get(crop, crop.capitalize())
                response += f"<strong>{crop_kn}:</strong><br>"
                response += f"• pH: {info['ph_range']}<br>"
                response += f"• ತಾಪಮಾನ: {info['temp_range']}<br>"
                response += f"• ಪ್ರದೇಶಗಳು: {info['regions']}<br><br>"
            
            response += "💡 <em>ಪ್ರತಿ ಬೆಳೆಗೆ ವಿವರವಾದ ಮಾಹಿತಿಗಾಗಿ ಪ್ರತ್ಯೇಕವಾಗಿ ಕೇಳಿ.</em>"
            
            suggestions = [
                f"{crop_translations.get(crops_found[0], crops_found[0].capitalize())} ಬಗ್ಗೆ ಹೇಳಿ",
                "ಲಾಭದಾಯಕತೆ ತುಲನೆ",
                "ಗೊಬ್ಬರ ಅವಶ್ಯಕತೆಗಳು"
            ]
        else:
            response = f"🌾 <strong>Crop Information</strong><br><br>"
            
            for crop in crops_found:
                info = CROP_INFO[crop]
                response += f"<strong>{crop.capitalize()}:</strong><br>"
                response += f"• pH: {info['ph_range']}<br>"
                response += f"• Temperature: {info['temp_range']}<br>"
                response += f"• Regions: {info['regions']}<br><br>"
            
            response += "💡 <em>Ask about each crop individually for detailed information.</em>"
            
            suggestions = [
                f"Tell me about {crops_found[0]}",
                "Compare profitability",
                "Fertilizer requirements"
            ]
        return response, suggestions
    
    # Recommendation queries
    if any(word in message for word in ['recommend', 'suggest', 'best', 'suitable', 'grow', 'plant', 'ಶಿಫಾರಸು', 'ಸೂಕ್ತ', 'ಬೆಳೆಯಿರಿ', 'ಬಿತ್ತಿರಿ']):
        if recommendations and len(recommendations) > 0:
            if language == 'kn':
                response = "🌾 <strong>ನಿಮ್ಮ ಪ್ರಮುಖ ಬೆಳೆ ಶಿಫಾರಸುಗಳು:</strong><br><br>"
                for i, rec in enumerate(recommendations[:3], 1):
                    crop_name = rec.get('crop', '')
                    crop_kn = crop_translations.get(crop_name.lower(), crop_name)
                    confidence = rec.get('confidence', 0)
                    response += f"{i}. <strong>{crop_kn}</strong> - {confidence}% ಹೊಂದಾಣಿಕೆ<br>"
                
                response += "<br>💡 ಇವು ನಿಮ್ಮ ಮಣ್ಣಿನ ನಿಯತಾಂಕಗಳು ಮತ್ತು ಸ್ಥಳದ ಆಧಾರದ ಮೇಲೆ!"
                
                top_crop_kn = crop_translations.get(recommendations[0].get('crop', '').lower(), recommendations[0].get('crop', 'ಪ್ರಮುಖ ಬೆಳೆ'))
                suggestions = [
                    f"{top_crop_kn} ಬಗ್ಗೆ ಹೇಳಿ",
                    "ಲಾಭದಾಯಕತೆ ತುಲನೆ ತೋರಿಸಿ",
                    "ನನಗೆ ಯಾವ ಗೊಬ್ಬರ ಬೇಕು?"
                ]
            else:
                response = "🌾 <strong>Your Top Crop Recommendations:</strong><br><br>"
                for i, rec in enumerate(recommendations[:3], 1):
                    crop_name = rec.get('crop', '')
                    confidence = rec.get('confidence', 0)
                    response += f"{i}. <strong>{crop_name}</strong> - {confidence}% match<br>"
                
                response += "<br>💡 These are based on your soil parameters and location!"
                
                suggestions = [
                    f"Tell me about {recommendations[0].get('crop', 'the top crop')}",
                    "Show profitability comparison",
                    "What fertilizer do I need?"
                ]
            return response, suggestions
        
        if language == 'kn':
            response = "🌾 <strong>ಬೆಳೆ ಶಿಫಾರಸು</strong><br><br>"
            response += "ನಿಮಗೆ ಉತ್ತಮ ಬೆಳೆಗಳನ್ನು ಶಿಫಾರಸು ಮಾಡಲು, ನನಗೆ ಬೇಕು:<br>"
            response += "• ನಿಮ್ಮ ಜಿಲ್ಲೆ/ಸ್ಥಳ<br>"
            response += "• ಮಣ್ಣಿನ NPK ಮಟ್ಟಗಳು<br>"
            response += "• ಮಣ್ಣಿನ pH ಮತ್ತು ತೇವಾಂಶ<br>"
            response += "• ತಾಪಮಾನ<br><br>"
            response += "ವೈಯಕ್ತಿಕ ಶಿಫಾರಸುಗಳನ್ನು ಪಡೆಯಲು ದಯವಿಟ್ಟು ಮುಖ್ಯ ಫಾರ್ಮ್ ಬಳಸಿ!"
            
            suggestions = [
                "ಪ್ರಸ್ತುತ ಋತು ಏನು?",
                "ಅತ್ಯಂತ ಲಾಭದಾಯಕ ಬೆಳೆಗಳು",
                "ಮಣ್ಣಿನ ಪರೀಕ್ಷೆಯ ಮಾಹಿತಿ"
            ]
        else:
            response = "🌾 <strong>Crop Recommendation</strong><br><br>"
            response += "To recommend the best crops for you, I need:<br>"
            response += "• Your district/location<br>"
            response += "• Soil NPK levels<br>"
            response += "• Soil pH and moisture<br>"
            response += "• Temperature<br><br>"
            response += "Please use the main form to get personalized recommendations!"
            
            suggestions = [
                "What is the current season?",
                "Most profitable crops",
                "Soil testing information"
            ]
        return response, suggestions
    
    # Comparison queries
    if any(word in message for word in ['compare', 'difference', 'versus', 'vs', 'better', 'ಹೋಲಿಸಿ', 'ವ್ಯತ್ಯಾಸ', 'ಉತ್ತಮ']):
        if language == 'kn':
            response = "⚖️ <strong>ಬೆಳೆ ತುಲನೆ</strong><br><br>"
            response += "ನಾನು ನಿಮಗೆ ಬೆಳೆಗಳನ್ನು ಹೋಲಿಸಲು ಸಹಾಯ ಮಾಡಬಲ್ಲೆ! ಪ್ರಯತ್ನಿಸಿ:<br>"
            response += "• 'ರಾಗಿ ಅಥವಾ ಹತ್ತಿ ಯಾವುದು ಹೆಚ್ಚು ಲಾಭದಾಯಕ?'<br>"
            response += "• 'ಕಾಫಿ ಮತ್ತು ಚಹಾದ ಗೊಬ್ಬರ ಅವಶ್ಯಕತೆಗಳನ್ನು ಹೋಲಿಸಿ'<br>"
            response += "• 'ನನ್ನ ಮಣ್ಣಿಗೆ ಯಾವುದು ಉತ್ತಮ?'<br><br>"
            response += "ಅಥವಾ ದೃಶ್ಯ ತುಲನೆಗಾಗಿ ವಿಶ್ಲೇಷಣೆ ಡ್ಯಾಶ್‌ಬೋರ್ಡ್‌ಗೆ ಭೇಟಿ ನೀಡಿ!"
            
            suggestions = [
                "ವಿಶ್ಲೇಷಣೆ ಡ್ಯಾಶ್‌ಬೋರ್ಡ್ ತೋರಿಸಿ",
                "ಅತ್ಯಂತ ಲಾಭದಾಯಕ ಬೆಳೆಗಳು",
                "ಮಣ್ಣಿನ ಅವಶ್ಯಕತೆಗಳ ತುಲನೆ"
            ]
        else:
            response = "⚖️ <strong>Crop Comparison</strong><br><br>"
            response += "I can help you compare crops! Try asking:<br>"
            response += "• 'Which is more profitable, Ragi or Cotton?'<br>"
            response += "• 'Compare fertilizer needs of Coffee and Tea'<br>"
            response += "• 'What's better for my soil?'<br><br>"
            response += "Or visit the Analytics Dashboard for visual comparisons!"
            
            suggestions = [
                "Show analytics dashboard",
                "Most profitable crops",
                "Soil requirements comparison"
            ]
        return response, suggestions
    
    # Website usage and instructions queries
    if any(word in message for word in ['how to use', 'how do i', 'instructions', 'guide', 'tutorial', 'update soil', 'enter data', 'input', 'parameters', 'ಹೇಗೆ ಬಳಸುವುದು', 'ಮಾರ್ಗದರ್ಶಿ', 'ಮಣ್ಣಿನ ಮಾಹಿತಿ ನಮೂದಿಸಿ', 'ಡೇಟಾ ನಮೂದಿಸಿ']):
        if language == 'kn':
            response = "📖 <strong>ವೆಬ್‌ಸೈಟ್ ಬಳಕೆಯ ಮಾರ್ಗದರ್ಶಿ</strong><br><br>"
            
            response += "🌾 <strong>ಬೆಳೆ ಶಿಫಾರಸು ಪಡೆಯಲು:</strong><br>"
            response += "1️⃣ ಮುಖಪುಟಕ್ಕೆ ಹೋಗಿ (ಮೇಲಿನ 'Krushi Mitra' ಲೋಗೋ ಕ್ಲಿಕ್ ಮಾಡಿ)<br>"
            response += "2️⃣ ನಿಮ್ಮ ಜಿಲ್ಲೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ<br>"
            response += "3️⃣ ಮಣ್ಣಿನ ನಿಯತಾಂಕಗಳನ್ನು ನಮೂದಿಸಿ:<br>"
            response += "   • <strong>N (ಸಾರಜನಕ):</strong> 0-300 kg/ha<br>"
            response += "   • <strong>P (ರಂಜಕ):</strong> 0-150 kg/ha<br>"
            response += "   • <strong>K (ಪೊಟ್ಯಾಸಿಯಮ್):</strong> 0-200 kg/ha<br>"
            response += "   • <strong>pH:</strong> 4.0-9.0 (ಆಮ್ಲೀಯತೆ/ಕ್ಷಾರೀಯತೆ)<br>"
            response += "   • <strong>ತೇವಾಂಶ:</strong> 0-100%<br>"
            response += "   • <strong>ತಾಪಮಾನ:</strong> 10-45°C<br>"
            response += "4️⃣ 'Get Recommendations' ಬಟನ್ ಕ್ಲಿಕ್ ಮಾಡಿ<br>"
            response += "5️⃣ ನಿಮ್ಮ ಮಣ್ಣಿಗೆ ಸೂಕ್ತವಾದ ಟಾಪ್ 3 ಬೆಳೆಗಳನ್ನು ನೋಡಿ!<br><br>"
            
            response += "📊 <strong>ವಿಶ್ಲೇಷಣೆ ವೀಕ್ಷಿಸಲು:</strong><br>"
            response += "• ಫಲಿತಾಂಶ ಪುಟದಲ್ಲಿ 'View Analytics Dashboard' ಕ್ಲಿಕ್ ಮಾಡಿ<br>"
            response += "• 8 ಸಂವಾದಾತ್ಮಕ ಚಾರ್ಟ್‌ಗಳನ್ನು ಅನ್ವೇಷಿಸಿ<br><br>"
            
            response += "🔍 <strong>ಬೆಳೆ ಮಾಹಿತಿ ಹುಡುಕಲು:</strong><br>"
            response += "• ಮುಖಪುಟದಲ್ಲಿ 'Crop Information Lookup' ಬಳಸಿ<br>"
            response += "• ಬೆಳೆಯ ಹೆಸರನ್ನು ಟೈಪ್ ಮಾಡಿ ಅಥವಾ ಆಯ್ಕೆಮಾಡಿ<br><br>"
            
            response += "💡 <strong>ಸಲಹೆ:</strong> ನಿಖರವಾದ NPK ಮೌಲ್ಯಗಳಿಗಾಗಿ ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ ಮಾಡಿಸಿ!"
            
            suggestions = [
                "ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ ಎಲ್ಲಿ ಮಾಡಿಸಬೇಕು?",
                "NPK ಎಂದರೇನು?",
                "ಲಾಭದಾಯಕ ಬೆಳೆಗಳು ತೋರಿಸಿ"
            ]
        else:
            response = "📖 <strong>Website Usage Guide</strong><br><br>"
            
            response += "🌾 <strong>To Get Crop Recommendations:</strong><br>"
            response += "1️⃣ Go to the Home Page (click 'Krushi Mitra' logo at top)<br>"
            response += "2️⃣ Select your district from the dropdown<br>"
            response += "3️⃣ Enter your soil parameters:<br>"
            response += "   • <strong>N (Nitrogen):</strong> 0-300 kg/ha<br>"
            response += "   • <strong>P (Phosphorus):</strong> 0-150 kg/ha<br>"
            response += "   • <strong>K (Potassium):</strong> 0-200 kg/ha<br>"
            response += "   • <strong>pH:</strong> 4.0-9.0 (acidity/alkalinity)<br>"
            response += "   • <strong>Moisture:</strong> 0-100%<br>"
            response += "   • <strong>Temperature:</strong> 10-45°C<br>"
            response += "4️⃣ Click 'Get Recommendations' button<br>"
            response += "5️⃣ View your top 3 recommended crops!<br><br>"
            
            response += "📊 <strong>To View Analytics:</strong><br>"
            response += "• Click 'View Analytics Dashboard' on results page<br>"
            response += "• Explore 8 interactive charts with crop data<br><br>"
            
            response += "🔍 <strong>To Look Up Crop Information:</strong><br>"
            response += "• Use 'Crop Information Lookup' on home page<br>"
            response += "• Type or select a crop name<br><br>"
            
            response += "💬 <strong>To Update Soil Parameters:</strong><br>"
            response += "• Return to home page and enter new values<br>"
            response += "• Get fresh recommendations instantly<br><br>"
            
            response += "💡 <strong>Tip:</strong> Get a soil test for accurate NPK values!"
            
            suggestions = [
                "Where to get soil test?",
                "What is NPK?",
                "Show profitable crops"
            ]
        return response, suggestions
    
    # Help/greeting
    if any(word in message for word in ['help', 'hello', 'hi', 'namaste', 'hey', 'start', 'ಸಹಾಯ', 'ನಮಸ್ಕಾರ', 'ಹಲೋ']):
        if language == 'kn':
            response = "🙏 <strong>ನಮಸ್ಕಾರ! ನಾನು ರೈತ ಸಹಾಯ</strong><br><br>"
            response += "ನಾನು ನಿಮಗೆ ಸಹಾಯ ಮಾಡಬಲ್ಲೆ:<br>"
            response += "• 🌦️ ಹವಾಮಾನ ಮತ್ತು ಋತುಮಾನ ಮಾಹಿತಿ<br>"
            response += "• 💰 ಲಾಭದಾಯಕತೆ ವಿಶ್ಲೇಷಣೆ<br>"
            response += "• 🌿 ಗೊಬ್ಬರ ಲೆಕ್ಕಾಚಾರಗಳು<br>"
            response += "• 🧪 ಮಣ್ಣಿನ ಅವಶ್ಯಕತೆಗಳು<br>"
            response += "• 🌾 ಬೆಳೆ ಶಿಫಾರಸುಗಳು<br>"
            response += "• 📖 ವೆಬ್‌ಸೈಟ್ ಬಳಕೆಯ ಮಾರ್ಗದರ್ಶನ<br><br>"
            response += "ಕರ್ನಾಟಕದಲ್ಲಿ ಕೃಷಿಯ ಬಗ್ಗೆ ನನಗೆ ಏನು ಬೇಕಾದರೂ ಕೇಳಿ!"
            
            suggestions = [
                "ವೆಬ್‌ಸೈಟ್ ಹೇಗೆ ಬಳಸುವುದು?",
                "ಹವಾಮಾನ ಹೇಗಿದೆ?",
                "ಲಾಭದಾಯಕ ಬೆಳೆಗಳು"
            ]
        else:
            response = "🙏 <strong>Namaste! I'm FarmerHelp</strong><br><br>"
            response += "I can assist you with:<br>"
            response += "• 🌦️ Weather & seasonal information<br>"
            response += "• 💰 Profitability analysis<br>"
            response += "• 🌿 Fertilizer calculations<br>"
            response += "• 🧪 Soil requirements<br>"
            response += "• 🌾 Crop recommendations<br>"
            response += "• 📖 Website usage guide<br><br>"
            response += "Just ask me anything about farming in Karnataka!"
            
            suggestions = [
                "How to use this website?",
                "What is the weather like?",
                "Show profitable crops"
            ]
        
        return response, suggestions
    
    # Default response
    if language == 'kn':
        response = "🤔 ಕ್ಷಮಿಸಿ, ನನಗೆ ಅರ್ಥವಾಗಲಿಲ್ಲ. ನಾನು ನಿಮಗೆ ಸಹಾಯ ಮಾಡಬಲ್ಲೆ:<br><br>"
        response += "• ಹವಾಮಾನ ಮತ್ತು ಋತುಮಾನ ಮಾಹಿತಿ<br>"
        response += "• ಬೆಳೆ ಲಾಭದಾಯಕತೆ ವಿಶ್ಲೇಷಣೆ<br>"
        response += "• ಗೊಬ್ಬರ ಅವಶ್ಯಕತೆಗಳು<br>"
        response += "• ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ ಮತ್ತು ಅವಶ್ಯಕತೆಗಳು<br>"
        response += "• ಬೆಳೆ ಶಿಫಾರಸುಗಳು<br><br>"
        response += "ಪ್ರಯತ್ನಿಸಿ: 'ಹವಾಮಾನ ಹೇಗಿದೆ?' ಅಥವಾ 'ಯಾವ ಬೆಳೆ ಹೆಚ್ಚು ಲಾಭದಾಯಕ?'"
    else:
        response = "🤔 I'm not sure I understood that. I can help you with:<br><br>"
        response += "• Weather and seasonal information<br>"
        response += "• Crop profitability analysis<br>"
        response += "• Fertilizer requirements<br>"
        response += "• Soil testing and requirements<br>"
        response += "• Crop recommendations<br><br>"
        response += "Try asking: 'What is the weather like?' or 'Which crop is most profitable?'"
    
    return response, default_suggestions


if __name__ == '__main__':
    app.run(debug=True)
