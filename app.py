import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from PIL import Image
import sys, os
from ultralytics import YOLO

fish_db = {
    "Tilapia (Nile & Red)": {
        "temp": (25, 30), "pH": (6.5, 9.0), "tan": (117, 233),
        "disp_temp": "25 - 30°C", "disp_pH": "6.5 - 9.0", "disp_nutrients": "117 - 233 mg/kg/day"
    },
    "Hito (Catfish)": {
        "temp": (27, 30), "pH": (6.5, 9.0), "tan": (240, 320),
        "disp_temp": "27 - 30°C", "disp_pH": "6.5 - 9.0", "disp_nutrients": "240 - 320 mg/kg/day"
    },
    "Karpa (Common Carp)": {
        "temp": (25, 30), "pH": (6.5, 9.0), "tan": (150, 250),
        "disp_temp": "25 - 30°C", "disp_pH": "6.5 - 9.0", "disp_nutrients": "150 - 250 mg/kg/day"
    },
    "Koi & Goldfish (Ornamental)": {
        "temp": (20, 28), "pH": (7.0, 8.5), "tan": (100, 200),
        "disp_temp": "20 - 28°C", "disp_pH": "7.0 - 8.5", "disp_nutrients": "100 - 200 mg/kg/day"
    }
}

plant_db = {
    "Pechay": {
        "temp": (22, 28), "pH": (6.0, 7.5), "n_acc": (17, 19),
        "disp_temp": "22 - 28°C", "disp_pH": "6.0 - 7.5", "disp_nutrients": "17 - 19 mg/day"
    },
    "Lettuce": {
        "temp": (20, 27), "pH": (6.0, 7.0), "n_acc": (9, 11),
        "disp_temp": "20 - 27°C", "disp_pH": "6.0 - 7.0", "disp_nutrients": "9 - 11 mg/day"
    },
    "Kangkong": {
        "temp": (25, 30), "pH": (6.0, 7.0), "n_acc": (20, 30),
        "disp_temp": "25 - 30°C", "disp_pH": "6.0 - 7.0", "disp_nutrients": "20 - 30 mg/day"
    },
    "Cabbage": {
        "temp": (20, 25), "pH": (6.0, 7.0), "n_acc": (45, 60),
        "disp_temp": "20 - 25°C", "disp_pH": "6.0 - 7.0", "disp_nutrients": "45 - 60 mg/day"
    }
}

@st.cache_resource
def build_fuzzy_system():
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

    rules = [
        ctrl.Rule(temp_match['low'] & ph_match['low'] & nutrient_ratio['low'], compatibility['very_low']),
        ctrl.Rule(temp_match['low'] & ph_match['low'] & nutrient_ratio['optimal'], compatibility['very_low']),
        ctrl.Rule(temp_match['low'] & ph_match['low'] & nutrient_ratio['high'], compatibility['very_low']),
        ctrl.Rule(temp_match['low'] & ph_match['medium'] & nutrient_ratio['low'], compatibility['low']),
        ctrl.Rule(temp_match['low'] & ph_match['medium'] & nutrient_ratio['optimal'], compatibility['low']),
        ctrl.Rule(temp_match['low'] & ph_match['medium'] & nutrient_ratio['high'], compatibility['very_low']),
        ctrl.Rule(temp_match['low'] & ph_match['high'] & nutrient_ratio['low'], compatibility['low']),
        ctrl.Rule(temp_match['low'] & ph_match['high'] & nutrient_ratio['optimal'], compatibility['moderate']),
        ctrl.Rule(temp_match['low'] & ph_match['high'] & nutrient_ratio['high'], compatibility['low']),
        ctrl.Rule(temp_match['medium'] & ph_match['low'] & nutrient_ratio['low'], compatibility['low']),
        ctrl.Rule(temp_match['medium'] & ph_match['low'] & nutrient_ratio['optimal'], compatibility['low']),
        ctrl.Rule(temp_match['medium'] & ph_match['low'] & nutrient_ratio['high'], compatibility['very_low']),
        ctrl.Rule(temp_match['medium'] & ph_match['medium'] & nutrient_ratio['low'], compatibility['low']),
        ctrl.Rule(temp_match['medium'] & ph_match['medium'] & nutrient_ratio['optimal'], compatibility['moderate']),
        ctrl.Rule(temp_match['medium'] & ph_match['medium'] & nutrient_ratio['high'], compatibility['low']),
        ctrl.Rule(temp_match['medium'] & ph_match['high'] & nutrient_ratio['low'], compatibility['moderate']),
        ctrl.Rule(temp_match['medium'] & ph_match['high'] & nutrient_ratio['optimal'], compatibility['high']),
        ctrl.Rule(temp_match['medium'] & ph_match['high'] & nutrient_ratio['high'], compatibility['moderate']),
        ctrl.Rule(temp_match['high'] & ph_match['low'] & nutrient_ratio['low'], compatibility['low']),
        ctrl.Rule(temp_match['high'] & ph_match['low'] & nutrient_ratio['optimal'], compatibility['moderate']),
        ctrl.Rule(temp_match['high'] & ph_match['low'] & nutrient_ratio['high'], compatibility['low']),
        ctrl.Rule(temp_match['high'] & ph_match['medium'] & nutrient_ratio['low'], compatibility['moderate']),
        ctrl.Rule(temp_match['high'] & ph_match['medium'] & nutrient_ratio['optimal'], compatibility['high']),
        ctrl.Rule(temp_match['high'] & ph_match['medium'] & nutrient_ratio['high'], compatibility['moderate']),
        ctrl.Rule(temp_match['high'] & ph_match['high'] & nutrient_ratio['low'], compatibility['moderate']),
        ctrl.Rule(temp_match['high'] & ph_match['high'] & nutrient_ratio['optimal'], compatibility['very_high']),
        ctrl.Rule(temp_match['high'] & ph_match['high'] & nutrient_ratio['high'], compatibility['moderate'])
    ]
    return ctrl.ControlSystem(rules)

compat_ctrl = build_fuzzy_system()

def get_base_path():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_vision_model():
    model_path = os.path.join(get_base_path(), 'yolo26n.pt')
    return YOLO(model_path)

# MATH HELPERS 
def calculate_match_percentage(range1, range2, is_ph=False):
    overlap = max(0, min(range1[1], range2[1]) - max(range1[0], range2[0]))
    if overlap == 0:
        return 0.0
    threshold = 0.5 if is_ph else 2.0
    match_score = (overlap / threshold) * 100
    return np.clip(match_score, 0, 100)

def evaluate_pairing(fish_data, plant_data):
    t_match = calculate_match_percentage(fish_data['temp'], plant_data['temp'])
    p_match = calculate_match_percentage(fish_data['pH'], plant_data['pH'], is_ph=True)
    ratio = (sum(fish_data['tan']) / 2.0) / (sum(plant_data['n_acc']) / 2.0)
    
    sim = ctrl.ControlSystemSimulation(compat_ctrl)
    sim.input['temp_match'] = np.clip(t_match, 0, 100)
    sim.input['ph_match'] = np.clip(p_match, 0, 100)
    sim.input['nutrient_ratio'] = np.clip(ratio, 0, 100)
    
    sim.compute()
    score = sim.output['compatibility']
    
    limiting_factor = None
    if score < 75: 
        issues = []
        if t_match < 50:
            issues.append("Temp mismatch")
        if p_match < 50:
            issues.append("pH mismatch")
        if ratio < 6:
            issues.append("Low nutrients (Fish produce too little TAN)")
        elif ratio > 20:
            issues.append("Excess nutrients (Risk of toxicity)")
            
        if issues:
            limiting_factor = " & ".join(issues)
            
    if score >= 75:
        tier = "🌟 Highly Recommended"
        color = "green"
    elif score >= 50:
        tier = "✅ Recommended"
        color = "blue"
    elif score >= 40:
        tier = "⚠️ Moderate"
        color = "orange"
    else:
        tier = "❌ Not Recommended"
        color = "red"
            
    return {
        "score": score,
        "limiting_factor": limiting_factor,
        "tier": tier,
        "color": color
    }

def display_plant_recommendations(target_name, target_data):
    st.markdown(f"### Target Profile: {target_name}")
    cols = st.columns(3)
    cols[0].metric("Water Temperature", target_data['disp_temp'])
    cols[1].metric("pH Level", target_data['disp_pH'])
    cols[2].metric("Nitrogen Accumulation", target_data['disp_nutrients'])
    
    st.markdown("---")
    st.subheader(f"Compatibility Rankings for {target_name}")
    
    results = []
    for f_name, f_data in fish_db.items():
        eval_result = evaluate_pairing(f_data, target_data)
        results.append((f_name, f_data, eval_result))
        
    results.sort(key=lambda x: x[2]["score"], reverse=True)
    
    card_cols = st.columns(len(results))
    for idx, (rec_name, rec_data, eval_data) in enumerate(results):
        with card_cols[idx]:
            with st.container(border=True): 
                st.markdown(f"#### 🐟 {rec_name}")
                st.progress(int(eval_data['score']) / 100, text=f"Compatibility: {eval_data['score']:.1f}%")
                
                st.markdown(f":{eval_data['color']}[**{eval_data['tier']}**]")
                
                if eval_data['limiting_factor']:
                    st.markdown(f":red[**Limiting Factor:** {eval_data['limiting_factor']}]")
                    
                st.write(f"**Temp:** {rec_data['disp_temp']}")
                st.write(f"**pH:** {rec_data['disp_pH']}")
                st.write(f"**TAN:** {rec_data['disp_nutrients']}")

    # --- FINAL RECOMMENDATION LOGIC ---
    st.markdown("---")
    st.subheader("💡 Final Recommendation")
    
    best_match_name = results[0][0]
    best_match_score = results[0][2]["score"]
    
    if best_match_score >= 40:
        st.success(f"Based on our analysis, we recommend pairing **{target_name}** with **{best_match_name}** (Compatibility: {best_match_score:.1f}%).")
    else:
        st.warning(f"There is currently no existing fish in the database that is compatible with **{target_name}**.")
        
    st.info("**How is this calculated?** The compatibility score uses a fuzzy logic system to evaluate the overlap in ideal **Water Temperature** and **pH Levels**, alongside the **Nutrient Ratio** (Total Ammonia Nitrogen produced by the fish vs. Nitrogen required by the plants). A higher percentage means the environments required for both species closely align.", icon="ℹ️")

st.set_page_config(page_title="PlantIsda", layout="wide")
st.title("🌿🐟 PlantIsda")
st.markdown("Find the perfect plant and fish combinations for aquaponics using PlantIsda!")

st.sidebar.header("Select Type")
selection_mode = st.sidebar.radio(
    "Choose how you want to start:",
    ["Select Manually", "Select by Image"]
)

st.markdown("---")

if selection_mode == "Select Manually":
    category = st.sidebar.selectbox("Choose a category:", ["Fish", "Plant"])

    if category == "Fish":
        st.subheader("Select a Target Fish")
        target_name = st.selectbox("Choose a fish species:", list(fish_db.keys()))
        
        if target_name:
            target_data = fish_db[target_name]
            st.markdown(f"### Target Profile: {target_name}")
            cols = st.columns(3)
            cols[0].metric("Water Temperature", target_data['disp_temp'])
            cols[1].metric("pH Level", target_data['disp_pH'])
            cols[2].metric("TAN Production Rate", target_data['disp_nutrients'])
            
            st.markdown("---")
            st.subheader(f"Compatibility Rankings for {target_name}")
            
            results = []
            for p_name, p_data in plant_db.items():
                eval_result = evaluate_pairing(target_data, p_data)
                results.append((p_name, p_data, eval_result))
                
            results.sort(key=lambda x: x[2]["score"], reverse=True)
            
            card_cols = st.columns(len(results))
            for idx, (rec_name, rec_data, eval_data) in enumerate(results):
                with card_cols[idx]:
                    with st.container(border=True):
                        st.markdown(f"#### 🌱 {rec_name}")
                        st.progress(int(eval_data['score']) / 100, text=f"Compatibility: {eval_data['score']:.1f}%")
                        
                        st.markdown(f":{eval_data['color']}[**{eval_data['tier']}**]")
                        
                        if eval_data['limiting_factor']:
                            st.markdown(f":red[**Limiting Factor:** {eval_data['limiting_factor']}]")
                            
                        st.write(f"**Temp:** {rec_data['disp_temp']}")
                        st.write(f"**pH:** {rec_data['disp_pH']}")
                        st.write(f"**N-Accumulation:** {rec_data['disp_nutrients']}")

            # --- FINAL RECOMMENDATION LOGIC ---
            st.markdown("---")
            st.subheader("💡 Final Recommendation")
            
            best_match_name = results[0][0]
            best_match_score = results[0][2]["score"]
            
            if best_match_score >= 40:
                st.success(f"Based on our analysis, we recommend pairing **{target_name}** with **{best_match_name}** (Compatibility: {best_match_score:.1f}%).")
            else:
                st.warning(f"There is currently no existing plant in the database that is compatible with **{target_name}**.")
                
            st.info("**How is this calculated?** The compatibility score uses a fuzzy logic system to evaluate the overlap in ideal **Water Temperature** and **pH Levels**, alongside the **Nutrient Ratio** (Total Ammonia Nitrogen produced by the fish vs. Nitrogen required by the plants). A higher percentage means the environments required for both species closely align.", icon="ℹ️")

    elif category == "Plant":
        st.subheader("Select a Target Plant")
        target_name = st.selectbox("Choose a plant species:", list(plant_db.keys()))
        if target_name:
            display_plant_recommendations(target_name, plant_db[target_name])

elif selection_mode == "Select by Image":
    st.subheader("📷 Plant Scanner")
    st.write("Upload a photo or use your camera to automatically detect a plant and evaluate compatibility.")
    
    yolo_to_db_mapping = {
        "cabbage": "Cabbage", "cabbage_fresh": "Cabbage", "cabbage_rotten": "Cabbage", "cabbage_wilted": "Cabbage",
        "kangkong": "Kangkong", "kangkong_fresh": "Kangkong", "kangkong_wilted": "Kangkong",
        "lettuce": "Lettuce", "lettuce_healthy": "Lettuce", "lettuce_unhealthy": "Lettuce",
        "pechay": "Pechay", "pechay_healthy": "Pechay", "pechay_unhealthy": "Pechay",
    }
    
    input_method = st.radio("Choose image source:", ["Upload File", "Take Picture"], horizontal=True)
    
    image_source = None
    if input_method == "Upload File":
        image_source = st.file_uploader("Select an image (PNG, JPG)", type=["png", "jpg", "jpeg"])
    elif input_method == "Take Picture":
        image_source = st.camera_input("Take a picture")
    
    if image_source is not None:
        try:
            model = load_vision_model()
            img = Image.open(image_source)
            
            if input_method == "Upload File":
                st.image(img, caption="Selected Image", width=400)
            
            with st.spinner("Running YOLO detection..."):
                results = model(img)
            
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes
                best_box = boxes[0] 
                
                conf = float(best_box.conf[0])
                class_id = int(best_box.cls[0])
                detected_class_raw = model.names[class_id].lower()
                
                target_name = yolo_to_db_mapping.get(detected_class_raw)
                st.success(f"**Detected:** {target_name}")
                
                if target_name and target_name in plant_db:
                    st.markdown("---")
                    display_plant_recommendations(target_name, plant_db[target_name])
                else:
                    st.warning(f"Detected a '{detected_class_raw}', but there are no matching parameters for it in the database right now.")
            else:
                st.error("No recognizable plants detected in the frame. Please try a clearer angle.")
                
        except Exception as e:
            st.error(f"Error loading or running the model. Ensure 'yolo26n.pt' is in the directory. Details: {e}")