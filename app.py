import os
import sys
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import streamlit as st
from PIL import Image
from ultralytics import YOLO

fish_db = {
    "Tilapia (Nile & Red)": {"temp": (25, 30), "pH": (6.5, 9.0), "tan": (117, 233), "unit": "mg/kg/day"},
    "Hito (Catfish)": {"temp": (27, 30), "pH": (6.5, 9.0), "tan": (240, 320), "unit": "mg/kg/day"},
    "Karpa (Common Carp)": {"temp": (25, 30), "pH": (150, 250), "tan": (150, 250), "unit": "mg/kg/day"}, 
    "Koi & Goldfish (Ornamental)": {"temp": (20, 28), "pH": (7.0, 8.5), "tan": (100, 200), "unit": "mg/kg/day"}
}

plant_db = {
    "Pechay": {"temp": (22, 28), "pH": (6.0, 7.5), "n_acc": (17, 19), "unit": "mg/day"},
    "Lettuce": {"temp": (20, 27), "pH": (6.0, 7.0), "n_acc": (9, 11), "unit": "mg/day"},
    "Kangkong": {"temp": (25, 30), "pH": (6.0, 7.0), "n_acc": (20, 30), "unit": "mg/day"},
    "Cabbage": {"temp": (20, 25), "pH": (6.0, 7.0), "n_acc": (45, 60), "unit": "mg/day"}
}

@st.cache_resource
def build_fuzzy():
    temp_match = ctrl.Antecedent(np.arange(0, 101, 1), 'temp_match')
    ph_match = ctrl.Antecedent(np.arange(0, 101, 1), 'ph_match')
    nutrient_ratio = ctrl.Antecedent(np.arange(0, 101, 1), 'nutrient_ratio')
    compatibility = ctrl.Consequent(np.arange(0, 101, 1), 'compatibility')

    for var in [temp_match, ph_match]:
        var['low'] = fuzz.trapmf(var.universe, [0, 0, 25, 45])
        var['medium'] = fuzz.trimf(var.universe, [25, 50, 75])
        var['high'] = fuzz.trapmf(var.universe, [55, 80, 100, 100])

    nutrient_ratio['low'] = fuzz.trapmf(nutrient_ratio.universe, [0, 0, 4, 8])
    nutrient_ratio['optimal'] = fuzz.trimf(nutrient_ratio.universe, [6, 12, 20])
    nutrient_ratio['high'] = fuzz.trapmf(nutrient_ratio.universe, [15, 25, 100, 100])

    compatibility['very_low'] = fuzz.trapmf(compatibility.universe, [0, 0, 15, 25])
    compatibility['low'] = fuzz.trimf(compatibility.universe, [15, 30, 45])
    compatibility['moderate'] = fuzz.trimf(compatibility.universe, [35, 50, 65])
    compatibility['high'] = fuzz.trimf(compatibility.universe, [55, 75, 85])
    compatibility['very_high'] = fuzz.trapmf(compatibility.universe, [75, 85, 100, 100])

    rule_map = {
        ('low', 'low'): {'low': 'very_low', 'optimal': 'very_low', 'high': 'very_low'},
        ('low', 'medium'): {'low': 'low', 'optimal': 'low', 'high': 'very_low'},
        ('low', 'high'): {'low': 'low', 'optimal': 'moderate', 'high': 'low'},
        ('medium', 'low'): {'low': 'low', 'optimal': 'low', 'high': 'very_low'},
        ('medium', 'medium'): {'low': 'low', 'optimal': 'moderate', 'high': 'low'},
        ('medium', 'high'): {'low': 'moderate', 'optimal': 'high', 'high': 'moderate'},
        ('high', 'low'): {'low': 'low', 'optimal': 'moderate', 'high': 'low'},
        ('high', 'medium'): {'low': 'moderate', 'optimal': 'high', 'high': 'moderate'},
        ('high', 'high'): {'low': 'moderate', 'optimal': 'very_high', 'high': 'moderate'}
    }

    rules = [
        ctrl.Rule(temp_match[t] & ph_match[p] & nutrient_ratio[n], compatibility[c])
        for (t, p), nut_dict in rule_map.items()
        for n, c in nut_dict.items()
    ]
    return ctrl.ControlSystem(rules)

compat_ctrl = build_fuzzy()

def get_base_path():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    return YOLO(os.path.join(get_base_path(), 'yolo26n.pt'))

def calculate_match_percentage(range1, range2, is_ph=False):
    overlap = max(0, min(range1[1], range2[1]) - max(range1[0], range2[0]))
    if overlap == 0:
        return 0.0
    threshold = 0.5 if is_ph else 2.0
    return np.clip((overlap / threshold) * 100, 0, 100)

def evaluate_pairing(fish_data, plant_data):
    t_match = calculate_match_percentage(fish_data['temp'], plant_data['temp'])
    p_match = calculate_match_percentage(fish_data['pH'], plant_data['pH'], is_ph=True)
    
    f_mid = sum(fish_data['tan']) / 2.0
    p_mid = sum(plant_data.get('n_acc', plant_data.get('tan'))) / 2.0
    ratio = f_mid / p_mid
    
    sim = ctrl.ControlSystemSimulation(compat_ctrl)
    sim.input['temp_match'] = np.clip(t_match, 0, 100)
    sim.input['ph_match'] = np.clip(p_match, 0, 100)
    sim.input['nutrient_ratio'] = np.clip(ratio, 0, 100)
    sim.compute()
    score = sim.output['compatibility']
    
    limiting_factor = None
    if score < 75: 
        issues = []
        if t_match < 50: issues.append("Temp mismatch")
        if p_match < 50: issues.append("pH mismatch")
        if ratio < 6: issues.append("Low nutrients (Fish produce too little TAN)")
        elif ratio > 20: issues.append("Excess nutrients (Risk of toxicity)")
        if issues: limiting_factor = " & ".join(issues)
            
    tiers = [(75, "Highly Recommended", "green"), (50, "Recommended", "blue"), (40, "Moderate", "orange"), (0, "Not Recommended", "red")]
    tier, color = next((t, c) for val, t, c in tiers if score >= val)
            
    return {"score": score, "limiting_factor": limiting_factor, "tier": tier, "color": color}

def get_disp_strings(profile, mode="fish"):
    nut_key = "tan" if mode == "fish" else "n_acc"
    return {
        "temp": f"{profile['temp'][0]} - {profile['temp'][1]}°C",
        "pH": f"{profile['pH'][0]} - {profile['pH'][1]}",
        "nutrients": f"{profile[nut_key][0]} - {profile[nut_key][1]} {profile['unit']}"
    }

def render_compatibility_view(target_name, is_target_fish=True):
    target_db, compare_db = (fish_db, plant_db) if is_target_fish else (plant_db, fish_db)
    target_data = target_db[target_name]
    t_str = get_disp_strings(target_data, "fish" if is_target_fish else "plant")
    
    st.markdown(f"### Target Profile: {target_name}")
    cols = st.columns(3)
    cols[0].metric("Water Temperature", t_str["temp"])
    cols[1].metric("pH Level", t_str["pH"])
    cols[2].metric("TAN Production Rate" if is_target_fish else "Nitrogen Accumulation", t_str["nutrients"])
    
    st.markdown("---")
    st.subheader(f"Compatibility Rankings for {target_name}")
    
    results = []
    for comp_name, comp_data in compare_db.items():
        eval_result = evaluate_pairing(target_data, comp_data) if is_target_fish else evaluate_pairing(comp_data, target_data)
        results.append((comp_name, comp_data, eval_result))
        
    results.sort(key=lambda x: x[2]["score"], reverse=True)
    card_cols = st.columns(len(results))
    
    for idx, (rec_name, rec_data, eval_data) in enumerate(results):
        with card_cols[idx]:
            with st.container(border=True): 
                st.markdown(f"#### {'🌱' if is_target_fish else '🐟'} {rec_name}")
                st.progress(int(eval_data['score']) / 100, text=f"Compatibility: {eval_data['score']:.1f}%")
                st.markdown(f":{eval_data['color']}[**{eval_data['tier']}**]")
                
                if eval_data['limiting_factor']:
                    st.markdown(f":red[**Limiting Factor:** {eval_data['limiting_factor']}]")
                    
                r_str = get_disp_strings(rec_data, "plant" if is_target_fish else "fish")
                st.write(f"**Temp:** {r_str['temp']}")
                st.write(f"**pH:** {r_str['pH']}")
                st.write(f"**{'N-Accumulation' if is_target_fish else 'TAN'}:** {r_str['nutrients']}")

    st.markdown("---")
    st.subheader("Recommendation")
    best_name, best_score = results[0][0], results[0][2]["score"]
    
    if best_score >= 40:
        st.success(f"Based on our analysis, we recommend pairing **{target_name}** with **{best_name}** (Compatibility: {best_score:.1f}%).")
    else:
        st.warning(f"There is currently no existing {'plant' if is_target_fish else 'fish'} in the database that is compatible with **{target_name}**.")
        
    st.info("**How is this calculated?** The compatibility score uses a fuzzy logic system to evaluate the overlap in ideal **Water Temperature** and **pH Levels**, alongside the **Nutrient Ratio** (Total Ammonia Nitrogen produced by the fish vs. Nitrogen required by the plants). A higher percentage means the environments required for both species closely align.", icon="ℹ️")

# --- App entry point ---
st.set_page_config(page_title="PlantIsda", layout="wide")
st.title("PlantIsda")
st.markdown("Find the perfect plant and fish combinations for aquaponics using PlantIsda!")

st.sidebar.header("Select Type")
selection_mode = st.sidebar.radio("Choose how you want to start:", ["Select Manually", "Select by Image"])
st.markdown("---")

if selection_mode == "Select Manually":
    category = st.sidebar.selectbox("Choose a category:", ["Fish", "Plant"])
    if category == "Fish":
        st.subheader("Select a Target Fish")
        target_name = st.selectbox("Choose a fish species:", list(fish_db.keys()))
        if target_name: render_compatibility_view(target_name, is_target_fish=True)
    elif category == "Plant":
        st.subheader("Select a Target Plant")
        target_name = st.selectbox("Choose a plant species:", list(plant_db.keys()))
        if target_name: render_compatibility_view(target_name, is_target_fish=False)

elif selection_mode == "Select by Image":
    st.subheader("Plant Scanner")
    st.write("Upload a photo or use your camera to automatically detect a plant and evaluate compatibility.")
    
    yolo_to_db_mapping = {
        "cabbage": "Cabbage", "cabbage_fresh": "Cabbage", "cabbage_rotten": "Cabbage", "cabbage_wilted": "Cabbage",
        "kangkong": "Kangkong", "kangkong_fresh": "Kangkong", "kangkong_wilted": "Kangkong",
        "lettuce": "Lettuce", "lettuce_healthy": "Lettuce", "lettuce_unhealthy": "Lettuce",
        "pechay": "Pechay", "pechay_healthy": "Pechay", "pechay_unhealthy": "Pechay",
    }
    
    input_method = st.radio("Choose image source:", ["Upload File", "Take Picture"], horizontal=True)
    image_source = st.file_uploader("Select an image (PNG, JPG)", type=["png", "jpg", "jpeg"]) if input_method == "Upload File" else st.camera_input("Take a picture")
    
    if image_source is not None:
        try:
            model = load_model()
            img = Image.open(image_source)
            if input_method == "Upload File":
                st.image(img, caption="Selected Image", width=400)
            
            with st.spinner("Running YOLO detection..."):
                results = model(img)
            
            if len(results[0].boxes) > 0:
                detected_class_raw = model.names[int(results[0].boxes[0].cls[0])].lower()
                target_name = yolo_to_db_mapping.get(detected_class_raw)
                st.success(f"**Detected:** {target_name}")
                
                if target_name and target_name in plant_db:
                    st.markdown("---")
                    render_compatibility_view(target_name, is_target_fish=False)
                else:
                    st.warning(f"Detected a '{detected_class_raw}', but there are no matching parameters for it in the database right now.")
            else:
                st.error("No recognizable plants detected in the frame. Please try a clearer angle.")
        except Exception as e:
            st.error(f"Error loading or running the model. Ensure 'yolo26n.pt' is in the directory. Details: {e}")