import pandas as pd
import numpy as np
import random
import os

def generate_patient_data(num_samples=1000):
    np.random.seed(42)
    random.seed(42)

    data = []
    
    complaints = ["right ocular pain", "right forearm burn", "arm pain, Lt", "ascites tapping", "distension, abd", "fever", "With chest discomfort"]
    diagnoses = ["Corneal abrasion", "Burn of hand, first degree dorsum", "Fracture of surgical neck of humerus", "Alcoholic liver cirrhosis", "Ascites", "Fever, unspecified", "Angina pectoris"]
    arrival_modes = ["car", "ambulance", "walking", "private_ambulance"]
    
    for _ in range(num_samples):
        # 0: Low, 1: Medium, 2: Critical
        priority = random.choices(["Low", "Medium", "Critical"], weights=[0.5, 0.3, 0.2])[0]
        
        # Base on priority to make data learnable
        age = random.randint(18, 90)
        hospital = random.choice([0, 1])
        sex = random.choice([0, 1])
        patient_num_hr = random.randint(1, 15)
        arrival_mode = random.choice(arrival_modes)
        injury = random.choice([0, 1])
        complaint = random.choice(complaints)
        mental = random.choice([0, 1])
        pain = random.choice([0, 1])
        pain_scale = random.randint(1, 10)
        
        # Vitals dependent on priority
        if priority == "Low":
            sbp = random.randint(110, 130)
            dbp = random.randint(70, 85)
            hr = random.randint(60, 100)
            rr = random.randint(12, 20)
            bt = round(random.uniform(36.5, 37.5), 1)
            saturation = random.randint(95, 100)
            ktas_rn = random.choice([4, 5])
            ktas_expert = random.choice([4, 5])
            result = random.choice([1, 2])
        elif priority == "Medium":
            sbp = random.randint(130, 150)
            dbp = random.randint(85, 95)
            hr = random.randint(100, 120)
            rr = random.randint(20, 25)
            bt = round(random.uniform(37.5, 39.0), 1)
            saturation = random.randint(90, 94)
            ktas_rn = random.choice([3])
            ktas_expert = random.choice([3])
            result = random.choice([2, 3])
        else: # Critical
            sbp = random.choice([random.randint(70, 90), random.randint(160, 200)])
            dbp = random.choice([random.randint(40, 60), random.randint(100, 120)])
            hr = random.choice([random.randint(40, 50), random.randint(120, 160)])
            rr = random.choice([random.randint(8, 10), random.randint(28, 40)])
            bt = round(random.choice([random.uniform(34.0, 35.5), random.uniform(39.5, 41.0)]), 1)
            saturation = random.randint(70, 89)
            ktas_rn = random.choice([1, 2])
            ktas_expert = random.choice([1, 2])
            result = random.choice([4, 5])
            
        diagnosis = random.choice(diagnoses)
        error_code = 0
        stay_min = random.randint(30, 600)
        ktas_duration_min = random.randint(1, 15)
        mistriage = random.choice([0, 1])

        data.append({
            "hospital": hospital,
            "sex": sex,
            "age": age,
            "patient_num_hr": patient_num_hr,
            "arrival_mode": arrival_mode,
            "injury": injury,
            "complaint": complaint,
            "mental": mental,
            "pain": pain,
            "pain_scale": pain_scale,
            "SBP": sbp,
            "DBP": dbp,
            "HR": hr,
            "RR": rr,
            "BT": bt,
            "saturation": saturation,
            "KTAS_RN": ktas_rn,
            "diagnosis": diagnosis,
            "result": result,
            "KTAS_expert": ktas_expert,
            "error_code": error_code,
            "stay_min": stay_min,
            "KTAS_duration_min": ktas_duration_min,
            "mistriage": mistriage,
            "Priority": priority
        })

    df = pd.DataFrame(data)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/synthetic_ehr_data.csv', index=False)
    print("Generated synthetic data with 24 features and saved to data/synthetic_ehr_data.csv")

if __name__ == "__main__":
    generate_patient_data()
