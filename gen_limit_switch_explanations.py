import sys, re, ast
sys.path.insert(0, r"C:\Users\Ravi\research\data-analysis")
from loader import load_dataframe
import pandas as pd
import os, datetime

df = load_dataframe(r"C:\Users\Ravi\Documents\Limit_Switches_attribute_value_change_report (4).xlsx")

# ── Domain knowledge ────────────────────────────────────────────────────────

HEAD_TYPE_DESC = {
    "Rotary Head":                 "actuated by rotational movement around an axis; used to detect the angular position of cams, shafts, and rotating machinery. The approach direction (side or top) is a distinct parameter captured separately as actuation_orientation.",
    "Plunger Head":                "actuated by a direct linear push force applied to a stem or piston; suited for detecting end-of-travel positions of cylinders, slides, and linear carriers. Approach direction is stored as actuation_orientation.",
    "Wobble Stick Head":           "omnidirectionally sensitive — the flexible stick deflects on contact from any lateral direction; deployed in conveyor systems and material handling where objects may approach from an unpredictable angle.",
    "Rod Lever Head":              "a rigid rod extending from the pivot arm that converts linear or near-linear motion into switch actuation; rod adjustability, material (fiberglass, stainless steel, carbon steel), and diameter are engineering-significant parameters.",
    "Roller Lever Head":           "a lever fitted with a rolling wheel at the tip to reduce friction and cam wear during repeated actuation; preferred in high-cycle applications where a sliding contact would cause accelerated mechanical wear.",
    "Adjustable Lever Head":       "a lever whose effective length or angle is field-adjustable to set the trip point without replacing the switch body.",
    "Spring Rod (Whisker) Head":   "a thin, flexible spring wire that deflects on the lightest contact force; provides the lowest operating force of any limit switch actuator type — used for detecting small, fragile, or light workpieces.",
    "Fork Lever Head":             "a bifurcated lever that straddles a moving part or guide rail; used in transfer line fixtures and pallet-detection applications.",
    "Offset Roller Lever Head":    "a roller lever offset from the switch centreline, allowing the switch body to be mounted away from the direct actuation path while maintaining correct trip geometry.",
    "Telescoping Rod Lever Head":  "a rod lever with adjustable length for fine trip-point tuning without additional hardware.",
}

ACTUATOR_TYPE_DESC = {
    "Pin Plunger":    "a short, small-diameter precision pin providing highly repeatable actuation at a fixed point with minimal side-load tolerance; preferred where installation space is severely constrained.",
    "Plunger":        "a standard cylindrical stem providing direct linear push actuation; the most common type for general industrial proximity and end-stop detection.",
    "Roller (Wheel)": "a wheel mounted on the plunger tip that rolls against cams or ramps, converting relative linear motion into switch actuation while minimising friction and wear on both the actuator tip and the cam surface.",
    "Lever":          "a pivoting arm providing mechanical advantage, allowing actuation by objects at a distance or by low-force contacts; lever length and adjustability are additional specification parameters.",
    "Wobble Stick":   "an omnidirectional flexible rod actuator; any lateral force in the operating plane triggers actuation — used in parts ejection and material handling.",
    "Cat Whisker":    "a very fine stiff wire providing the lowest available actuation force; used to detect the lightest workpieces or thin stock material without marking or deflecting the part.",
    "Roller Lever":   "a lever with a roller tip; preferred over a plain lever wherever repeated cam-follower contact would otherwise cause mechanical wear on the lever tip.",
    "Rod Lever":      "a long rigid rod lever for detecting objects at greater reach distances; rod diameter and material govern bending stiffness, adjustability range, and corrosion resistance.",
}

CHANGE_TYPE_RATIONALE = {
    "Value Standardization": (
        "The original value embedded additional specification details — such as orientation, material, or adjustability — directly inside the attribute field as free-text. "
        "Industry practice and IEC/ISO catalogue standards define a single canonical name per head or actuator type; "
        "additional physical properties must be modelled as separate discrete attributes. "
        "Embedding them in the value string makes faceted filtering, cross-catalogue comparison, and automated data validation unreliable."
    ),
    "Abbreviation Standardization": (
        "The original value was a bare number without a unit symbol, which is inherently ambiguous — the same numeral could represent volts, amps, or hertz depending on context. "
        "IEC 60947 mandates that all electrical ratings include the SI unit symbol. "
        "Abbreviated values that omit units cannot be ingested reliably into PIM systems that enforce unit compliance or into electronic catalogue exchange formats such as BMEcat and ETIM."
    ),
    "Vocabulary Normalization": (
        "The original value used an informal or abbreviated form that omitted the measurement unit. "
        "Normalisation appends the SI unit to produce a fully qualified, machine-readable value consistent with IEC 60947 marking requirements and electronic catalogue exchange schemas."
    ),
    "Attribute Split": (
        "The original value combined multiple independent properties — such as contact type and count, or multiple IP codes — into a single delimited string. "
        "These are distinct, independently queryable properties. "
        "Combining them in one field prevents independent filtering, breaks parametric search, and makes compliance validation impossible without string parsing."
    ),
    "Synonym Resolution": (
        "The original value was a colloquial, abbreviated, or manufacturer-specific synonym not aligned with preferred industry terminology. "
        "Multiple synonyms for the same concept prevent data aggregation, break cross-reference lookups, and inflate the apparent variety in catalogue filter lists."
    ),
}

def parse_features(feat_str):
    if not feat_str or str(feat_str) in ("nan", "None", ""):
        return {}
    try:
        val = ast.literal_eval(str(feat_str))
        if isinstance(val, dict):
            return val
    except Exception:
        pass
    result = {}
    for pair in re.split(r",\s*(?=[a-z_]+\s*:)", str(feat_str)):
        kv = pair.split(":", 1)
        if len(kv) == 2:
            result[kv[0].strip()] = kv[1].strip()
    return result


def explain_head_type(orig, norm, feat_str, ct):
    feats = parse_features(feat_str)
    rationale = CHANGE_TYPE_RATIONALE.get(ct, "")
    head_desc = HEAD_TYPE_DESC.get(norm, "a limit switch head actuated in a manner characteristic of its name")

    feat_parts = []
    if feats.get("actuation_orientation"):
        feat_parts.append(f"actuation orientation ({feats['actuation_orientation']}) is retained as a discrete attribute")
    if feats.get("lever_adjustability"):
        feat_parts.append(f"lever adjustability ({feats['lever_adjustability']}) is modelled as a separate Boolean or enum attribute")
    if feats.get("rod_diameter"):
        feat_parts.append(f"rod diameter ({feats['rod_diameter']}) is stored as a numeric dimensional fact")
    if feats.get("rod_material"):
        feat_parts.append(f"rod material ({feats['rod_material']}) is captured in a dedicated material attribute")
    if feats.get("roller_material"):
        feat_parts.append(f"roller material ({feats['roller_material']}) is retained to preserve corrosion-resistance information")
    if feats.get("main_material"):
        feat_parts.append(f"body material ({feats['main_material']}) is captured in a material attribute")
    if feats.get("bearing_type"):
        feat_parts.append(f"bearing type ({feats['bearing_type']}) is stored as a mechanical sub-attribute")
    if feats.get("free_text"):
        feat_parts.append(f"the unresolved descriptor \"{feats['free_text']}\" is preserved in a supplementary notes field for manual review")

    feat_sentence = ""
    if feat_parts:
        feat_sentence = (
            " To preserve specification completeness without data loss, "
            + "; ".join(feat_parts) + "."
        )

    return (
        f"Technical: The {norm} is {head_desc} "
        f"The original value \"{orig}\" conflated the head-type classification with additional specification detail in a single string. "
        f"{rationale}{feat_sentence} "
        f"PIM/MDM: Head Type must be a controlled-vocabulary attribute populated from a closed canonical list aligned with IEC 60947-5-1 and ETIM class EC000024 (Limit Switch). "
        f"Compound descriptors must not appear as attribute values; each physical characteristic — orientation, material, adjustability — belongs in its own dedicated attribute with its own unit and pick-list. "
        f"User benefit: Engineers searching for a {norm} now find all matching products regardless of how orientation or material was worded in the original catalogue entry, "
        f"eliminating missed results caused by synonym proliferation and enabling reliable multi-facet filtering by head geometry, material, and adjustability independently."
    )


def explain_actuator_type(orig, norm, feat_str, ct):
    feats = parse_features(feat_str)
    rationale = CHANGE_TYPE_RATIONALE.get(ct, "")

    if not norm or norm.strip() in ("nan", "None", ""):
        return (
            f"Technical: The original value \"{orig}\" indicates the absence of a factory-fitted actuator — the switch is designed to be operated directly by a customer-supplied mechanism. "
            f"Cataloguing this as a non-null text string is incorrect; the absence of an actuator is properly represented as a null or empty value in the Actuator Type attribute, not as a type name. "
            f"PIM/MDM: Actuator Type should be left blank (null) when no actuator is fitted. Storing \"No Actuator\" as a pick-list entry pollutes the controlled vocabulary with a pseudo-type, "
            f"causes filter-logic failures when users select specific actuator types, and misrepresents the product configuration in parametric search. "
            f"A dedicated Boolean attribute \"Actuator Included\" is the correct modelling pattern for this condition. "
            f"User benefit: Catalogue filters for specific actuator types correctly exclude actuator-less base switches. "
            f"Engineers who explicitly need a base switch (to fit a custom actuator) can use an \"Actuator Included = No\" filter without encountering misleading text values."
        )

    act_desc = ACTUATOR_TYPE_DESC.get(norm, "a standard limit switch actuating mechanism")

    # Context-appropriate rationale: Abbreviation Standardization for actuator types
    # means shortened name, not missing units — override the generic rationale
    if ct == "Abbreviation Standardization":
        rationale = (
            f"The original value \"{orig}\" was an abbreviated shorthand that omits the qualifier words "
            f"which distinguish this actuator type from related types in the same family. "
            f"Abbreviated type names cause ambiguity in catalogue search, prevent exact-match filtering, "
            f"and fail electronic catalogue exchange schemas that require the full canonical designation."
        )

    feat_parts = []
    if feats.get("adjustability"):
        feat_parts.append(f"adjustability ({feats['adjustability']}) is stored as a separate discrete attribute")
    if feats.get("rod_diameter"):
        feat_parts.append(f"rod diameter ({feats['rod_diameter']}) is stored as a numeric dimension")
    if feats.get("rod_material"):
        feat_parts.append(f"rod material ({feats['rod_material']}) is captured in a material attribute")
    if feats.get("roller_material"):
        feat_parts.append(f"roller material ({feats['roller_material']}) is retained as a material sub-attribute")
    if feats.get("lever_length_mm"):
        feat_parts.append(f"lever length ({feats['lever_length_mm']} mm) is modelled as a numeric dimensional attribute")

    feat_sentence = (
        " Specification details extracted from the original string — "
        + "; ".join(feat_parts)
        + " — are preserved in dedicated attributes without data loss."
    ) if feat_parts else ""

    return (
        f"Technical: The {norm} is {act_desc} "
        f"The original value \"{orig}\" was a {ct.lower()} of the standard term or embedded specification details that belong in discrete attributes. "
        f"{rationale}{feat_sentence} "
        f"PIM/MDM: Actuator Type must draw from a closed ETIM-aligned pick-list (EC000024, attribute EF000004). "
        f"Adjustability, material, and dimensional properties must be modelled as independent attributes, "
        f"enabling engineers to filter on each axis without parsing compound strings. "
        f"User benefit: Procurement engineers and system integrators can now combine Actuator Type with material, adjustability, and dimensional filters as independent facets, "
        f"producing precise shortlists from a single search operation. Cross-supplier comparison becomes meaningful because all suppliers use the same canonical actuator vocabulary."
    )


def explain_voltage_rating(orig, norm, feat_str, ct):
    volt_match = re.match(r"([\d.]+)\s*V", str(norm))
    volt_val = volt_match.group(1) if volt_match else norm

    current_type = "unspecified AC or DC"
    if "current_type" in str(feat_str):
        ct_match = re.search(r"current_type.*?:\s*['\"]([^'\"]+)['\"]", str(feat_str))
        if ct_match and ct_match.group(1) not in ("Unspecified", ""):
            current_type = ct_match.group(1)

    return (
        f"Technical: The voltage rating defines the maximum continuous working voltage the switch contacts are certified to handle without insulation breakdown or destructive arcing between open contacts. "
        f"A rating of {volt_val} V means the switch is approved for circuits operating at up to {volt_val} volts ({current_type}). "
        f"IEC 60947-5-1 (Low-voltage switchgear, Part 5-1: Control circuit devices) and UL 508 both require voltage ratings to include the SI unit symbol 'V' alongside the numeric value. "
        f"The original value \"{orig}\" omitted the unit symbol, making it ambiguous in isolation and non-compliant with electronic catalogue exchange standards (BMEcat, ETIM, GS1 Data Quality Framework). "
        f"PIM/MDM: Voltage Rating must be stored as a structured attribute pairing a numeric value with a unit field ('V'), "
        f"or as a fully qualified string conforming to ISO 31-1 notation. "
        f"Systems that ingest bare numerals risk unit-confusion errors when aggregating multi-supplier data. "
        f"The additional_features metadata records current type and frequency for further enrichment into separate AC Voltage Rating and DC Voltage Rating attributes. "
        f"User benefit: Control circuit engineers can filter by voltage level with confidence — a search for '240 V' returns only 240-volt-rated switches, "
        f"not entries where 240 is an ambiguous numeral. Automated BOM validators can compare switch ratings against circuit operating voltages without building unit-inference logic."
    )


def explain_current_rating(orig, norm, feat_str, ct):
    amp_match = re.match(r"([\d.]+)\s*A", str(norm))
    amp_val = amp_match.group(1) if amp_match else norm

    return (
        f"Technical: The current rating specifies the maximum continuous current the switch contacts can conduct and interrupt without excessive temperature rise, contact welding, or accelerated erosion. "
        f"A {amp_val} A rating governs the maximum resistive or inductive load the switch can control directly; exceeding it causes contact damage, false tripping, and premature mechanical failure. "
        f"IEC 60947-5-1 Clause 4 mandates that current ratings be expressed in amperes with the unit symbol 'A'; bare numeric values are non-compliant with the standard and with electronic catalogue protocols. "
        f"The original value \"{orig}\" lacked the unit symbol, making it machine-ambiguous. "
        f"PIM/MDM: Current Rating must be stored with its SI unit explicit — either as part of the value string or in a paired unit field. "
        f"Where an IEC utilisation category is available (AC-15 for AC solenoid loads, DC-13 for DC solenoid loads per IEC 60947-5-1 Table 2), "
        f"it must be stored as a separate structured attribute to enable compliance-level filtering. "
        f"User benefit: Electrical designers can filter results by minimum current rating to guarantee the switch handles their load safely. "
        f"Automated parametric selection tools can apply current rating as a hard constraint, preventing under-rated switches from appearing in shortlists and removing the need for manual datasheet verification."
    )


def explain_contact_form(orig, norm, feat_str, ct):
    feats = parse_features(feat_str)
    nc = str(feats.get("nc_count", "0"))
    no = str(feats.get("no_count", "0"))
    co = str(feats.get("co_count", "0"))
    arrangement = feats.get("arrangement", "independent")
    pos_open = str(feats.get("positive_opening", "False")).lower() == "true"

    contact_parts = []
    if nc not in ("", "0"): contact_parts.append(f"{nc} normally-closed (NC) contact(s)")
    if no not in ("", "0"): contact_parts.append(f"{no} normally-open (NO) contact(s)")
    if co not in ("", "0"): contact_parts.append(f"{co} changeover (CO/SPDT) contact(s)")
    contact_desc = " and ".join(contact_parts) if contact_parts else "the described contact arrangement"

    pos_note = (
        " This configuration includes positive-opening action per IEC 60947-5-1 Annex K, "
        "which is mandatory for use in safety-related control circuits where contact welding must be mechanically detected."
    ) if pos_open else ""

    return (
        f"Technical: The contact form defines the electrical switching topology of the limit switch output stage. "
        f"This configuration provides {contact_desc} with {str(arrangement).lower()} circuit arrangement.{pos_note} "
        f"NC contacts are closed in the rest position and open when the actuator is operated — used in safety stop and de-energise-on-trip circuits. "
        f"NO contacts are open at rest and close on actuation — used in signal and enable circuits. "
        f"CO contacts provide both NC and NO functions from a single common terminal (SPDT switching). "
        f"The original notation \"{orig}\" used a spaced, written-out format inconsistent with compact IEC-standard contact-form notation. "
        f"{CHANGE_TYPE_RATIONALE['Attribute Split']} "
        f"PIM/MDM: Contact Form must be stored in compact IEC notation (e.g. 1NC+1NO, 1CO, 2NC) as the primary value. "
        f"Additionally, the constituent properties — NC count, NO count, CO count, circuit arrangement, and positive-opening flag — "
        f"must be stored as separate Boolean and integer attributes to enable granular multi-condition filtering. "
        f"User benefit: Control engineers can filter by exact contact form to find switches compatible with their circuit topology in a single query. "
        f"Safety circuit designers can filter specifically on positive-opening action without reading individual datasheets. "
        f"CAE and ECAD tools that auto-generate circuit symbols from part library data can parse structured contact-form attributes directly."
    )


def explain_termination(orig, norm, feat_str, ct):
    feats = parse_features(feat_str)
    term_descs = {
        "Screw Terminals": (
            "field-wireable screw-clamp or screw-cage connections that the installer completes on-site; "
            "provides maximum flexibility for conductor gauge, circuit reconfiguration, and field repair"
        ),
        "Threaded Conduit Entry": (
            "a threaded entry port in the switch housing accepting a conduit fitting, "
            "allowing pre-run wiring to enter the enclosure directly. "
            "Thread standard (NPT or PG) and thread size determine which conduit fittings are mechanically compatible"
        ),
        "Prewired Cable": (
            "a factory-installed cable pre-terminated to the switch contacts; "
            "no field wiring is required, eliminating termination errors and reducing installation time. "
            "Cable certification (UL, CSA, CE) governs the permitted application environment"
        ),
        "Other / Unspecified": (
            "a termination type that could not be resolved to a recognised standard designation from the original value; "
            "the original text is preserved in additional_features for manual reclassification"
        ),
    }
    term_desc = term_descs.get(norm, "a wiring connection method defined by the normalised value")

    feat_parts = []
    if feats.get("conduit_entry_thread_standard"):
        feat_parts.append(
            f"thread standard ({feats['conduit_entry_thread_standard']}) and thread size "
            f"({feats.get('conduit_entry_thread_size', 'unspecified')}) are extracted as filterable child attributes"
        )
    if feats.get("cable_certification"):
        feat_parts.append(f"cable certification ({feats['cable_certification']}) is stored as a compliance attribute")
    if feats.get("raw_unparsed"):
        feat_parts.append(f"the unresolved original text (\"{feats['raw_unparsed']}\") is retained in a notes field for manual review")

    feat_sentence = (" " + "; ".join(feat_parts).capitalize() + ".") if feat_parts else ""

    return (
        f"Technical: The termination type defines how the electrical connection between the limit switch and the machine wiring is made. "
        f"This switch uses {term_desc}.{feat_sentence} "
        f"The original value \"{orig}\" was a colloquial, abbreviated, or non-standard synonym for the preferred catalogue term. "
        f"{CHANGE_TYPE_RATIONALE['Synonym Resolution']} "
        f"PIM/MDM: Termination must draw from a standardised pick-list covering Screw Terminals, Spring-Cage Terminals, Prewired Cable, Threaded Conduit Entry (with thread standard sub-attribute), and connector variants (M12, M8). "
        f"For Threaded Conduit Entry, thread standard (NPT, PG, M-metric) and thread size must be stored as child attributes to allow compatibility filtering against installed conduit infrastructure. "
        f"User benefit: Maintenance engineers can filter replacement switches by termination type to match existing site wiring without incurring rewiring cost or extended downtime. "
        f"E-procurement systems can enforce site-standard termination types as a mandatory filter before returning supplier results."
    )


def explain_ip_rating(orig, norm, feat_str, ct):
    ip_codes = re.findall(r"IP\d+", str(norm))
    codes_str = " and ".join(ip_codes) if ip_codes else norm

    ip_meanings = {
        "IP65": "dust-tight (first digit 6) and protected against sustained low-pressure water jets from any direction (second digit 5) per IEC 60529",
        "IP66": "dust-tight (first digit 6) and protected against powerful water jets from any direction (second digit 6) per IEC 60529",
        "IP67": "dust-tight (first digit 6) and protected against temporary full immersion up to 1 m for 30 minutes (second digit 7) per IEC 60529",
        "IP68": "dust-tight (first digit 6) and protected against continuous immersion beyond 1 m at conditions specified by the manufacturer (second digit 8) per IEC 60529",
    }
    meanings = "; ".join(
        f"{c}: {ip_meanings.get(c, 'a defined ingress-protection level per IEC 60529')}"
        for c in ip_codes
    ) if ip_codes else "the specified ingress-protection level"

    if ct == "Attribute Split":
        reason = (
            f"The original value \"{orig}\" encoded two IP codes in a pipe-delimited string, indicating the switch holds dual certification. "
            f"A single-value text field cannot represent a multi-valued attribute; storing multiple ratings in one string breaks equality filtering and prevents independent querying of each rating. "
            f"{CHANGE_TYPE_RATIONALE['Attribute Split']}"
        )
        pim_note = (
            "IP Rating must be modelled as a multi-value (list) attribute to correctly represent switches certified to more than one IP level. "
            "Single-value fields with embedded delimiters are a common data-modelling error that prevents correct faceted search, "
            "range-comparison filtering (e.g. IP67 or higher), and compliance validation."
        )
    else:
        reason = (
            f"The original value \"{orig}\" used a bare IP code string without normalised list formatting consistent with structured catalogue models. "
            f"{CHANGE_TYPE_RATIONALE['Abbreviation Standardization']}"
        )
        pim_note = (
            "IP Rating should be stored as a structured list attribute to uniformly accommodate both single and multi-rated products. "
            "Consistent list representation enables correct greater-than-or-equal filter logic "
            "(e.g. 'show all products rated IP65 or higher') across the entire catalogue."
        )

    return (
        f"Technical: The IP (Ingress Protection) rating per IEC 60529 classifies the switch enclosure's resistance to solid particle and liquid ingress using a two-digit code. "
        f"This switch carries rating(s) {codes_str}: {meanings}. "
        f"In limit switch applications, IP rating determines suitability for washdown environments (food and beverage, pharmaceutical), outdoor installations, and cutting-fluid exposure in machine tools. "
        f"Selecting an insufficient IP level causes water ingress into the contact chamber, accelerated corrosion, insulation failure, and premature switch failure. "
        f"{reason} "
        f"PIM/MDM: {pim_note} "
        f"User benefit: Design engineers can filter precisely for the minimum IP level mandated by their environmental specification (e.g. IP67 for full-immersion wash zones) without free-text searches that miss pipe-delimited multi-rating entries. "
        f"Compliance and quality teams can generate accurate reports of all switch variants meeting a minimum IP threshold for zone-classified or ATEX-adjacent hazardous-location applications."
    )


def generate_explanation(attribute, original, normalized, additional_features, change_type):
    if attribute == "Head Type":
        return explain_head_type(original, normalized, additional_features, change_type)
    elif attribute == "Actuator Type":
        return explain_actuator_type(original, normalized, additional_features, change_type)
    elif attribute == "Voltage Rating":
        return explain_voltage_rating(original, normalized, additional_features, change_type)
    elif attribute == "Current Rating":
        return explain_current_rating(original, normalized, additional_features, change_type)
    elif attribute == "Contact Form":
        return explain_contact_form(original, normalized, additional_features, change_type)
    elif attribute == "Termination":
        return explain_termination(original, normalized, additional_features, change_type)
    elif attribute == "IP Rating":
        return explain_ip_rating(original, normalized, additional_features, change_type)
    else:
        return (
            f"Value \"{original}\" normalised to \"{normalized}\" via {change_type.lower()} "
            f"to align with catalogue standards and improve data consistency."
        )


output_rows = []
for _, row in df.iterrows():
    attr  = str(row.get("Attribute", ""))
    orig  = str(row.get("Original Value", ""))
    norm  = str(row.get("Normalized Value", ""))
    feats = str(row.get("Additional Features", "")) if str(row.get("Additional Features", "")) != "nan" else ""
    ct    = str(row.get("Change Type", ""))
    rec   = row.get("Records Affected", "")

    output_rows.append({
        "Attribute":          attr,
        "Original Value":     orig,
        "Normalized Value":   norm,
        "Additional Features":feats,
        "Change Type":        ct,
        "Records Affected":   rec,
        "Explanation":        generate_explanation(attr, orig, norm, feats, ct),
    })

out_df = pd.DataFrame(output_rows)
ts = datetime.date.today().strftime("%Y%m%d")
out_path = rf"C:\Users\Ravi\Documents\Limit_Switches_normalized_explanations_{ts}.csv"
out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"Saved: {out_path}  ({len(out_df)} rows)")

# Spot-check one row per attribute
for attr in out_df["Attribute"].unique():
    s = out_df[out_df["Attribute"] == attr].iloc[0]
    print(f"\n{'='*70}")
    print(f"[{attr}] ({s['Change Type']})")
    print(f"  '{s['Original Value']}' -> '{s['Normalized Value']}'")
    print(f"  {s['Explanation'][:500]}...")
