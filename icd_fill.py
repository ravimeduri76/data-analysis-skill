import sys, pandas as pd
sys.path.insert(0, r"C:\Users\Ravi\research\data-analysis")
from loader import save_csv

ICD_MAP = {
    # A
    "A fib":                    "I48.91",
    "A Fib":                    "I48.91",
    "Afib":                     "I48.91",
    "AFib":                     "I48.91",
    "AFIB":                     "I48.91",
    "Chronic Afib":             "I48.20",
    "Acute Hepatitis C reaction":"B17.10",
    "Advanced dementia":        "F03.90",
    "Alzeimer's Disease":       "G30.9",
    "Alzheimer's Disease":      "G30.9",
    "Anemia":                   "D64.9",
    "Anemia, Unspecified":      "D64.9",
    "Anemia, unspecified":      "D64.9",
    "anxiety":                  "F41.9",
    "Anxiety":                  "F41.9",
    "Occasional Anxiety":       "F41.9",
    "Aortoiliac Atherosclerotic Disease": "I70.0",
    "Arthritis":                "M13.9",
    "arthritis":                "M13.9",
    "arthritis ":               "M13.9",
    "Asthma":                   "J45.909",
    "Atherosclerotic heart disease with unstable angina": "I25.110",
    "Autoimmune Deficiency in nerves": "G62.9",
    # B
    "back pain":                "M54.50",
    "Benign essential hypertension": "I10",
    "Bilateral AKA":            "Z89.619",
    "bilateral hearing loss":   "H91.93",
    "Bipolar":                  "F31.9",
    "bladder cancer":           "C67.9",
    "Bladder spasm":            "N32.89",
    "Bladder/Bowel incontinence": "R32",
    "Blind R eye":              "H54.61",
    "Blind to Right Eye":       "H54.61",
    "Blindness":                "H54.7",
    "Blindness right eye":      "H54.61",
    "BPH":                      "N40.0",
    "BPH without outflow obstruction": "N40.0",
    # C
    "C7A.093=Malignant carcinoid tumor of the kidney": "C7A.093",
    "CAD":                      "I25.10",
    "CAD, A-Feb, HTN, arthritis, incontinence": None,
    "Cancer Mesothelioma":      "C45.9",
    "Cellulitis":               "L03.90",
    "CELLULITIS":               "L03.90",
    "Cerebral Infarct":         "I63.9",
    "Cerebral infarction":      "I63.9",
    "Cerebral vascular accident":"I63.9",
    "cervicalgia":              "M54.2",
    "Charcot's joint of foot":  "M14.679",
    "CHF":                      "I50.9",
    "CHF, COPD, LBP L Sciatica, Incontinent": None,
    "Chronic Afib":             "I48.20",
    "Chronic back pain":        "M54.50",
    "Chronic kidney disease":   "N18.9",
    "Chronic Kidney disease":   "N18.9",
    "Chronic kidney disease stage 3": "N18.30",
    "chronic pain":             "G89.29",
    "Chronic Pain":             "G89.29",
    "Chronic pain":             "G89.29",
    "Chronic Pain Syndrome":    "G89.29",
    "Chronic Pulmonary Fibrosis":"J84.10",
    "Chronic Regional Pain Syndrome": "G90.50",
    "Chronic Venous Hypertention with Ulcers": "I87.319",
    "Cirrhosis of liver":       "K74.60",
    "Cirrhosis of the liver":   "K74.60",
    "CKD":                      "N18.9",
    "CLOGGED TEAR DUCTS":       "H04.559",
    "Colitis":                  "K52.9",
    "CONGESTIVE HEART FAILURE": "I50.9",
    "Congestive heart failure":  "I50.9",
    "Congestive Heart Failure":  "I50.9",
    "Congestive Heart Failure Unspecified": "I50.9",
    "constipation":             "K59.00",
    "Constipation":             "K59.00",
    "COPD":                     "J44.1",
    "Copd":                     "J44.1",
    "Coronary Artery Disease":  "I25.10",
    "coronary atherosclerosis": "I25.10",
    "cough":                    "R05.9",
    "CREST (Raynaud's, esophageal GERD, Scleroderma, Tactilitic)": "M34.1",
    "CVA":                      "I63.9",
    "CVA, DM2, HTN, neuropathy, vertigo, arthritis": None,
    # D
    "D3A090=Benign carcinoid tumor of the bronchus and lung": "D3A.090",
    "DDD":                      "M51.36",
    "Debility":                 "R53.81",
    "Degenerative Disc Disease":"M51.36",
    "Degenerative disc disease":"M51.36",
    "Degenerative disk disease":"M51.36",
    "Degenerative Disk Disease - Chronic back pain": "M51.36",
    "Dementia":                 "F03.90",
    "Depression":               "F32.9",
    "Diabet":                   "E11.9",
    "Diabetes":                 "E11.9",
    "DIABETES":                 "E11.9",
    "Diabetes Mellitus":        "E11.9",
    "Diabetes Mellulitis":      "E11.9",
    "Diabetes type 2":          "E11.9",
    "Diabetic":                 "E11.9",
    "diabetic neuropathy":      "E11.40",
    "Diabetic peripheral neuropathy": "E11.40",
    "Diabetic retinopathy":     "E11.319",
    "Dialysis":                 "Z99.2",
    "diastolic dysfunction":    "I50.30",
    "DJD":                      "M19.90",
    "DM":                       "E11.9",
    "DM type 2":                "E11.9",
    "DMII":                     "E11.9",
    # E
    "Early stages of dementia": "G31.84",
    "eczema":                   "L30.9",
    "Edema":                    "R60.9",
    "Elevated PSA":             "R97.20",
    "enlarged prostate":        "N40.0",
    "Enlarged Prostate":        "N40.0",
    "Enlarged prostate":        "N40.0",
    "Espohageal Varices":       "I85.00",
    "Essential HTN":            "I10",
    "Exposure to potentially hazardous substance": "Z77.098",
    # F
    "Fall risk, tires easily, hypertension": None,
    "Fracture right shoulder":  "S42.201A",
    # G
    "gallbladder Cancer":       "C23",
    "Generalized weakness":     "R53.1",
    "GERD":                     "K21.0",
    "glaucoma":                 "H40.9",
    "Glaucoma":                 "H40.9",
    "Gout":                     "M10.9",
    "GOUT":                     "M10.9",
    # H
    "Hard of Hearing":          "H91.90",
    "Headaches":                "R51.9",
    "Hearing Loss":             "H91.90",
    "Hearing loss":             "H91.90",
    "hearing loss":             "H91.90",
    "Heart Arhythmia":          "I49.9",
    "Heart Arrythmia":          "I49.9",
    "Heart Disease":            "I51.9",
    "heart disease":            "I51.9",
    "HEART DISEASE":            "I51.9",
    "hemiparesis as late effect of CVA": "I69.359",
    "Hernia":                   "K46.9",
    "hernia repair":            "Z87.198",
    "Herpes":                   "B00.9",
    "HF":                       "I50.9",
    "Hiatal Hernia":            "K44.9",
    "high cholesterol":         "E78.00",
    "history prostate":         "Z85.46",
    "HLD":                      "E78.5",
    "HTN":                      "I10",
    "HTN, Hearing Loss, Arthritis, Bilateral bunions, Incontinent": None,
    "Hx anemia":                "D64.9",
    "Hx bowel obstruction surgery 10/2025": "Z87.198",
    "Hx CABG":                  "Z95.1",
    "Hx fall with fx lt kneecap":"Z87.39",
    "HX of Carcinoma of bas of tongue": "Z85.818",
    "Hx of Cataract Surgery":   "Z98.41",
    "hx of CVA":                "Z86.73",
    "HX of CVA x 3":            "Z86.73",
    "Hx of Gallbladder removal":"Z90.49",
    "Hx of Migraines":          "G43.909",
    "HX of Polio":              "B91",
    "Hx of Polio":              "B91",
    "HX of shingles":           "Z86.19",
    "Hx of stents in RLE.":     "Z95.820",
    "HX of Stroke (2019)":      "Z86.73",
    "Hx of Stroke (2019)":      "Z86.73",
    "HX of trigeminal Neuralgia":"G50.0",
    "Hx of UTI":                "Z87.440",
    "Hx of UTI's":              "Z87.440",
    "Hx pneumonia":             "Z87.01",
    "Hx right leg fracture":    "Z87.39",
    "Hypercholesterolemia":     "E78.00",
    "Hyperlipidemia":           "E78.5",
    "hyperlipidemia":           "E78.5",
    "Hypertension":             "I10",
    "hypertension":             "I10",
    "Hyperthyroidism":          "E05.90",
    "Hyperuricemia":            "E79.0",
    "Hypotension":              "I95.9",
    "Hypothyroidism":           "E03.9",
    "Hysterectomy":             "Z90.710",
    # I
    "I5.9=Heart failure, unspecified": "I50.9",
    "IDDM":                     "E10.9",
    "Impaired hearing":         "H91.90",
    "Impaired vision":          "H54.7",
    "Insomnia":                 "G47.00",
    "insomnia":                 "G47.00",
    "intracranial meningioma":  "D32.0",
    # K
    "Kidney Carcinoma":         "C64.9",
    # L
    "L1 spinal fracture":       "S32.001A",
    "Left knee pain":           "M25.562",
    "Left knee replacement x2": "Z96.652",
    "Left Mastectomy":          "Z90.12",
    "Legally blind":            "H54.8",
    "Liver Cancer":             "C22.9",
    "lumbar spondylosis":       "M47.816",
    "Lung cancer":              "C34.90",
    # M
    "M R54 Age-Related Physical Debility": "R54",
    "Macular degeneration":     "H35.30",
    "Major depression disorder, DM 11": None,
    "Major Depressive Order with reoccurring sever psychotic symptoms":   "F33.3",
    "Major Depressive Order with reoccurring Severe Psychotic Symptoms":  "F33.3",
    "Male erectile disorder":   "N52.9",
    "male hypogonadism":        "E29.1",
    "Memory deficits":          "R41.3",
    "metabolic syndrome X":     "E88.81",
    "MI x 2":                   "I25.2",
    "migraines":                "G43.909",
    "Mild Cognitive impairment of uncertain  or unknown": "G31.84",
    "Mood Disorder d/t phsyolo cond w major depressive":  "F06.31",
    "Mood disorder d/t physolo cond w major depressive":  "F06.31",
    "MS":                       "G35",
    "multiple joint pain":      "M25.50",
    "Myocardial Infarction":    "I25.2",
    # N
    "Neoplasm of uncertain behavior of lung": "D38.1",
    "Neurogenic bladder":       "N31.9",
    "neuropathy":               "G62.9",
    "Neuropathy":               "G62.9",
    "NIDDM":                    "E11.9",
    # O
    "OA":                       "M19.90",
    "OA, OP, Afib, R side paralysis. CVA, HTN, neuropathy, GERD, L Mastectomy, weakness": None,
    "Osteoarthritis":           "M19.90",
    "osteoarthritis":           "M19.90",
    "Osteoarthrosis":           "M19.90",
    "OSTEOPOROSIS":             "M81.0",
    "Osteoporosis":             "M81.0",
    "other disorders of peripheral nervous system": "G64",
    "Otits media":              "H66.90",
    "Oxygen dependent":         "Z99.81",
    # P
    "Pacemaker/ Defib":         "Z95.0",
    "Pacemaker/Defib":          "Z95.0",
    "Pacer":                    "Z95.0",
    "PAD":                      "I73.9",
    "pain":                     "R52",
    "Pain":                     "R52",
    "PAIN":                     "R52",
    "Pain Disorder":            "G89.29",
    "Pain disorder":            "G89.29",
    "panic disorder":           "F41.0",
    "Parkinson's Disease":      "G20",
    "Parkinsons":               "G20",
    "PARKINSONS":               "G20",
    "Poor Balance":             "R26.81",
    "post traumatic":           "F43.10",
    "Prostate Cancer":          "C61",
    "Prostate cancer":          "C61",
    "PROSTATE CANCER":          "C61",
    "PTSD":                     "F43.10",
    "ptsd":                     "F43.10",
    "PTSd":                     "F43.10",
    "PVD":                      "I73.9",
    # R
    "R Side weakness":          "G81.90",
    "R sided weakness":         "G81.90",
    "RA":                       "M06.9",
    "Radiculopathy":            "M54.10",
    "Recurrent Depressive Disorder": "F33.9",
    "RHEUMATOID ARTHRITIS":     "M06.9",
    "Rheumatoid Arthritis":     "M06.9",
    "right aboue the knee amputation": "Z89.611",
    "RIGHT ABOVE KNEE AMPUTATION":     "Z89.611",
    "Right hip pain":           "M25.551",
    "Right Hip Pain":           "M25.551",
    "Right S/P Lobectomy":      "Z90.2",
    "Right Sciatic Nerve Pain": "M54.41",
    "RIGHT SIDED PARALYSIS":    "G81.90",
    "right ventricular systolic dysfunction": "I50.810",
    # S
    "Schizophrenia":            "F20.9",
    "seborrheic dermatitis":    "L21.9",
    "Seizures":                 "R56.9",
    "Shortness of breath":      "R06.00",
    "Single Heart Bypass":      "Z95.1",
    "Skin Cancer":              "C44.91",
    "Skin cancer":              "C44.91",
    "sleep apnea":              "G47.30",
    "Sleep Apnea":              "G47.30",
    "Sleep apnea":              "G47.30",
    "Spinal Chord Injury":      "G95.9",
    "Stroke":                   "I63.9",
    # T
    "tachycardia":              "R00.0",
    "Thyroid CX":               "C73",
    "Thyroidectomy":            "Z90.09",
    "TIA":                      "G45.9",
    "Tinnitus":                 "H93.19",
    "tinnitus, bilateral":      "H93.13",
    "Tonsils and adnoid removal": "Z87.890",
    "traumatic injury d/t assault": "T14.90XA",
    "Tremors":                  "R25.1",
    "tremors":                  "R25.1",
    "Type 2 diabetes":          "E11.9",
    "Type 2 DM":                "E11.9",
    "Type II Diabetes":         "E11.9",
    # V
    "Vertigo":                  "R42",
    "Vitamin B12 deficiency anemia": "D51.9",
    # W
    "weakness":                 "R53.1",
    "Weakness":                 "R53.1",
    "Weakness in BLE r/t back pain": "M54.50",
}

# Also handle the garbled Parkinson encoding
PARKINSON_RE = "Parkinson"

df = pd.read_csv(r"C:\Users\Ravi\Elara_Probus\Probus_Data\Client_Medical_History_corrected.csv")
col_d = "diagnosis"
null_before = df["corrected diagnosis code"].isna().sum()

def fill_code(row):
    if pd.notna(row["corrected diagnosis code"]):
        return row["corrected diagnosis code"]
    desc = str(row[col_d]).strip() if pd.notna(row[col_d]) else None
    if not desc or desc == "0":
        return None
    # Exact match first
    if desc in ICD_MAP:
        return ICD_MAP[desc]
    # Garbled Parkinson variant (encoding artefact)
    if desc.startswith("Parkinson") and len(desc) < 20:
        return "G20"
    # Multi-condition heuristic: 3+ comma-separated items -> None
    if desc.count(",") >= 2:
        return None
    return None

df["corrected diagnosis code"] = df.apply(fill_code, axis=1)

total      = len(df)
filled     = df["corrected diagnosis code"].notna().sum()
still_null = df["corrected diagnosis code"].isna().sum()
newly_filled = null_before - still_null

print(f"Total rows                     : {total:,}")
print(f"Codes now filled               : {filled:,}  ({filled/total*100:.1f}%)")
print(f"Newly filled from lookup table : {newly_filled:,}")
print(f"Remaining null                 : {still_null:,}  ({still_null/total*100:.1f}%)")

# Show any residual nulls
residual = df.loc[
    df["corrected diagnosis code"].isna() &
    df[col_d].notna() &
    (df[col_d].str.strip() != "0"),
    col_d
].unique()
if len(residual):
    print(f"\nResidual unmatched ({len(residual)}):")
    for v in sorted(residual, key=lambda x: str(x).lower()):
        print(f"  {repr(v)}")

out_path = save_csv(df, r"C:\Users\Ravi\Elara_Probus\Probus_Data\Client_Medical_History_corrected.csv")
print(f"\nSaved: {out_path}")
