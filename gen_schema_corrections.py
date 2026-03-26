# gen_schema_corrections.py
# Reads Limit_Switches_schema_recommendation_report.xlsx and produces a corrected CSV with:
#   Column C (Change Type): corrected for consistency with A vs B and DT changes
#   Column J (Better Explanation): new expert explanations ordered by
#       functional/technical reason > industry standard > PIM/MDM practice
#
# Run:  python gen_schema_corrections.py
# Out:  Documents/Limit_Switches_schema_corrected_<date>.csv

import sys, os, datetime
sys.path.insert(0, '.')
from loader import load_dataframe, save_csv
import pandas as pd

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df = load_dataframe(r'C:\Users\Ravi\Documents\Limit_Switches_schema_recommendation_report.xlsx')
df = df.copy()

def g(row, col):
    v = row.get(col, '')
    return str(v).strip() if pd.notna(v) else ''

# ---------------------------------------------------------------------------
# Corrected Change Types  (keyed by 0-based row index)
# Introduced categories: "Data Type Change", "Rename + Data Type Change"
# Fixed: wrongly-Rename rows where name is unchanged, wrong Attribute Split rows,
#        Rename rows that are truly Attribute Merges (multiple temps → one target)
# ---------------------------------------------------------------------------
CT_OVERRIDES = {
    0:   "Rename + Data Type Change",   # Torque → Terminal tightening torque; Constant→Float
    1:   "Data Type Change",            # Torque UOM unchanged; Constant→Text
    3:   "Rename + Data Type Change",   # Port Style → Port Type/Thread Type; Constant→Text
    10:  "Data Type Change",            # Mounting Hole Size unchanged; Text→Float
    11:  "No Change",                   # Mounting Hole Size UOM unchanged; Text→Text (was Rename)
    12:  "Data Type Change",            # Control Voltage unchanged; Constant→Float
    14:  "Attribute Merge",             # Temperature Range Operating Ambient → Ambient op. temp range (same target as rows 33/81/124/150/155)
    17:  "Data Type Change",            # Roller Diameter unchanged; Constant→Float
    22:  "Data Type Change",            # Measurement Range unchanged; Constant→Text
    31:  "Data Type Change",            # Supply Voltage unchanged; Text→Float
    36:  "Data Type Change",            # Maximum Pressure Measured unchanged; Constant→Float
    38:  "Data Type Change",            # Operating Pressure unchanged; Constant→Float
    45:  "Rename + Data Type Change",   # Wire Size → Wire size range; Constant→Float
    47:  "Rename + Data Type Change",   # Lever → Actuator type; Constant→Text
    48:  "Data Type Change",            # Pitch unchanged; Constant→Float
    51:  "Data Type Change",            # Shock Resistance unchanged; Constant→Text
    55:  "Data Type Change",            # Number of Contacts unchanged; Constant→Integer
    60:  "Rename + Data Type Change",   # Number of Poles → Number of poles / contact circuits; Float→Integer
    62:  "Rename + Data Type Change",   # Size → Overall dimensions; Constant→Text
    65:  "No Change",                   # Weight UOM unchanged; Text→Text (was Rename)
    69:  "Data Type Change",            # Actuator Material unchanged; Constant→Text
    71:  "Rename + Data Type Change",   # Mechanical Life → Mechanical life (endurance); Text→Float
    73:  "Data Type Change",            # Switching Voltage unchanged; Constant→Float
    77:  "Rename + Data Type Change",   # Speed → Maximum switching frequency; Text→Float
    81:  "Attribute Merge",             # Temperature Operating Range → Ambient op. temp range (was wrongly Attribute Split)
    83:  "Rename + Data Type Change",   # Contact Type → Contact operation type; Constant→Text
    84:  "Data Type Change",            # Power unchanged; Constant→Float
    91:  "Data Type Change",            # Number of Pins unchanged; Constant→Integer
    95:  "Data Type Change",            # Number of Conductors unchanged; Constant→Integer
    96:  "No Change",                   # Type Port unchanged; Text→Text (was Rename)
    97:  "Data Type Change",            # Base Material unchanged; Constant→Text (was Rename)
    104: "Data Type Change",            # Seal Type unchanged; Constant→Boolean (was Rename)
    105: "Rename + Data Type Change",   # Accuracy → Repeat accuracy; Text→Float
    107: "Rename + Data Type Change",   # Humidity → Humidity range (non-condensing); Constant→Text
    113: "Rename + Data Type Change",   # Operating Force → Operating force (OF); Text→Float
    115: "Data Type Change",            # Warranty unchanged; Text→Float
    120: "Rename + Data Type Change",   # DC Voltage → NEMA DC voltage rating; Constant→Float
    124: "Attribute Merge",             # Maximum Operating Temperature → Ambient op. temp range (same target as rows 33/81/150)
    127: "Data Type Change",            # Frequency unchanged; Constant→Integer
    129: "Data Type Change",            # Overall Length unchanged; Constant→Float
    136: "Rename + Data Type Change",   # Voltage Rating → Rated operational voltage (Ue); Text→Float
    140: "Data Type Change",            # Number of Levels unchanged; Constant→Integer
    141: "Rename + Data Type Change",   # Release Force → Releasing force (RF); Text→Float
    143: "Data Type Change",            # Number of Decks unchanged; Constant→Integer
    144: "Data Type Change",            # Number of Positions unchanged; Float→Integer
    147: "Data Type Change",            # Input Voltage unchanged; Text→Float
    153: "Data Type Change",            # Hole Size unchanged; Text→Float
    155: "Attribute Merge",             # Minimum Operating Temperature → Ambient op. temp range (same target as rows 33/81/124/150)
}

# ---------------------------------------------------------------------------
# Better Explanations  (keyed by 0-based row index)
# Structure: functional/technical reason | industry standard | PIM/MDM practice
# ---------------------------------------------------------------------------
EXPLANATIONS = {
    # ---- Termination torque ----
    0: ("Terminal tightening torque specifies the maximum fastening torque for wire terminals "
        "on the switch body, preventing mechanical damage and ensuring reliable electrical "
        "connection under vibration. The original name 'Torque' is ambiguous — it could refer "
        "to actuator torque, spring torque, or cam torque. 'Terminal tightening torque' matches "
        "IEC 60947-5-1 and manufacturer datasheet terminology. "
        "Changing from Constant to Float is required because torque is a continuous physical "
        "quantity (typically 0.2–2.5 N·m) that must support range queries, unit conversion, "
        "and tool selection. PIM: enables filterable spec comparison across supplier catalogs."),

    1: ("The companion UOM attribute must store a controlled unit string such as 'N·m'. "
        "The original Constant type implies a single fixed value per SKU; Text is more "
        "appropriate here because the unit label may vary by region ('N·m' vs 'lbf·in') and "
        "PIM systems need to store and display it alongside the numeric torque value. "
        "Note: the attribute name should ideally align with Row 0 as 'Terminal tightening torque UOM'."),

    2: ("IP Rating (Ingress Protection) is already the widely recognised shorthand for the "
        "IEC 60529 protection code and requires no renaming. "
        "The regex pattern ^(IP\\d\\d[A-Z]?)$ enforces correct format (e.g., IP67, IP65W), "
        "preventing free-text entries such as 'IP-67' or 'IP 67' that break downstream filtering. "
        "PIM: validated format enables faceted search by protection level across all product families."),

    3: ("The conduit or cable entry interface determines which cable glands, conduit fittings, "
        "and sealing accessories are mechanically compatible with the switch body. "
        "'Port Style' is a non-standard internal label; 'Port Type / Thread Type' describes the "
        "physical standard (e.g., NPT 1/2\", PG11, M20x1.5) used for conduit interface. "
        "Changing from Constant to Text allows mixed thread standards (NPT, PG, metric) within "
        "the same attribute. IEC 60947-5-1 and ETIM EC000024 use thread type as a distinct "
        "attribute. PIM: enables filtering by conduit standard, critical for panel builder BOM selection."),

    4: ("'Wire Entry' and 'Cable Entry' refer to the same physical feature: the opening where "
        "cable or conduit connects to the switch housing. Renaming to 'Cable Entry / Wire Entry' "
        "makes both search terms findable as synonyms. IEC 60947-5-1 uses 'cable entry' as the "
        "standard term. PIM: a single canonical name eliminates duplicate attribute columns and "
        "ensures consistent cross-reference between wiring accessories and switch bodies."),

    5: ("Roller Material specifies the contact surface material of the roller actuator tip "
        "(e.g., stainless steel, nylon, polyamide). This directly affects wear rate, chemical "
        "resistance, and permissible cam speed. Name and type already match the canonical "
        "schema; no change is needed. ETIM EC000024 includes roller material as a standard "
        "attribute for cam-operated limit switches. PIM: enables material-based filtering "
        "for corrosive or food-grade environments."),

    6: ("NEMA-rated AC voltage is the supply voltage specification under NEMA ICS 14 / NEMA ICS 2 "
        "standards, distinct from IEC Ue ratings. Many North American products list NEMA voltage "
        "alongside IEC ratings as separate attributes. Renaming from 'AC Voltage' to "
        "'NEMA AC voltage rating' disambiguates the standard context and prevents incorrect "
        "mapping to IEC Ue attributes. Float type is unchanged; values are numeric (e.g., 120, 240). "
        "PIM: enables correct filtering for UL/NEMA-listed products in North American configurators."),

    7: ("The companion UOM for NEMA AC voltage is renamed to maintain alignment with the parent "
        "attribute 'NEMA AC voltage rating'. Constant type is retained as the unit is fixed (V). "
        "PIM: consistent naming of value–UOM pairs is mandatory for unit-aware PIM systems "
        "(e.g., Akeneo, Salsify) that display paired attributes."),

    8: ("Actuator type is the primary mechanical interface classification of a limit switch "
        "(e.g., pin plunger, roller lever, adjustable rod). It is the single most important "
        "search and filtering attribute for specifying the correct switch for a cam-driven "
        "application. Name and type already match the ETIM EC000024 canonical attribute; "
        "no change is needed. PIM: used as a top-level facet filter in product selectors."),

    9: ("Input Type categorises the signal input method (e.g., mechanical, inductive, "
        "photoelectric). Name and type are already consistent with the canonical schema. "
        "Constant type (controlled vocabulary) is appropriate. No change required."),

    10: ("Mounting Hole Size is a numeric dimension (e.g., 4.5 mm, 0.177 in) that specifies "
         "the screw or bolt clearance diameter for panel mounting. Storing it as Text prevents "
         "numeric range queries and unit conversion. Changing to Float enables dimensional "
         "filtering (e.g., 'show all switches with mounting hole <= 5 mm'). "
         "A companion UOM attribute (Row 11) stores the unit. "
         "PIM: Float type is mandatory for any dimension used in mechanical compatibility checks."),

    11: ("'Mounting Hole Size UOM' is unchanged — the name matches the source and no data type "
         "change is applicable (Text→Text). The previous Change Type 'Rename' was incorrect. "
         "The UOM attribute stores the unit string for the mounting hole dimension (e.g., 'mm', 'in'). "
         "Text is the correct type for a UOM label. No change required."),

    12: ("Control Voltage is the coil/solenoid voltage used to energise auxiliary circuits "
         "or indicator lights attached to the switch. It is a numeric value (e.g., 24 V DC, "
         "230 V AC) requiring Float type for range-based filtering and derating calculations. "
         "The Constant type implies a fixed controlled-vocabulary value, which incorrectly "
         "prevents multi-value or range storage. PIM: Float enables filtering by supply "
         "voltage in electromechanical BOM configurators."),

    13: ("The UOM companion to Control Voltage is correctly typed as Constant (fixed to 'V'). "
         "No name or type change required."),

    14: ("Multiple source attributes — 'Temperature Range Operating Ambient' (Row 14), "
         "'Temperature Rating' (Row 33), 'Temperature Operating Range' (Row 81), "
         "'Maximum Operating Temperature' (Row 124), 'Operating Temperature' (Row 150), "
         "and 'Minimum Operating Temperature' (Row 155) — all describe the same physical "
         "property: the ambient temperature range within which the switch operates reliably. "
         "IEC 60947-5-1 cl. 7.2.2 specifies ambient operating temperature as a single range "
         "(e.g., -25°C to +70°C). 'Ambient operating temperature range' is the ETIM EC000024 "
         "canonical term. Merging eliminates redundant columns and conflicting values. "
         "Data type changes from Constant to Text to store range notation (e.g., '-25...+70°C'). "
         "PIM: a single canonical attribute enables consistent temperature-based filtering "
         "across all product families."),

    15: ("The UOM companion for Temperature Range Operating Ambient is unchanged "
         "(Constant, stores °C or °F). No correction needed."),

    16: ("Package Type describes the product form factor or package style for ordering and "
         "logistics purposes. Name and type are consistent with the schema. No change required."),

    17: ("Roller Diameter is a physical measurement (e.g., 18 mm, 30 mm) that determines "
         "the cam lift required to operate the switch and affects mechanical advantage. "
         "Constant type prevents numeric filtering; Float enables range queries "
         "(e.g., 'show rollers with diameter 16–20 mm'). "
         "ETIM EC000024 specifies roller diameter as a Float with companion UOM (Row 18). "
         "PIM: numeric type is required for mechanical design tools that check cam-to-roller compatibility."),

    18: ("Roller Diameter UOM is correctly typed as Constant (fixed unit, typically 'mm'). "
         "No change required."),

    19: ("Special Features captures optional certifications or design variants "
         "(e.g., 'gold-plated contacts', 'slow-action mechanism') not covered by standard "
         "attributes. Text type is appropriate for free-form feature descriptions. "
         "No change required. PIM: searchable as a keyword field for specialist requirements."),

    20: ("'Material', 'Body Material' (Row 26), 'Casing Material' (Row 58), and "
         "'Housing Material' (Row 145) all describe the primary enclosure material "
         "(e.g., PA66, die-cast zinc, stainless steel). IEC 60947-5-1 and ETIM EC000024 "
         "use a single 'Housing material' attribute. Merging eliminates conflicting entries "
         "across columns for the same property. "
         "PIM: one canonical material attribute enables cross-family comparison, "
         "chemical resistance filtering, and RoHS/REACH compliance tagging."),

    21: ("Communication lists the fieldbus or IO-Link protocol if applicable. "
         "Name and type match the schema. No change required."),

    22: ("Measurement Range stores the sensor detection range (relevant for inductive or "
         "magnetic variants). Constant type implies a fixed single value per SKU, but range "
         "values such as '5–10 mm' require Text for proper storage and display. "
         "Changing to Text accurately represents the range notation used in datasheets "
         "and avoids loss of the hyphenated format. PIM: Text enables pattern-based search "
         "and display of sensor operating ranges in product listings."),

    23: ("Measurement Range UOM stores the unit for the detection range (e.g., 'mm'). "
         "Constant type is appropriate. No change required."),

    24: ("Compatibility lists compatible product families or accessories. "
         "Name and type match the schema. No change required."),

    25: ("'Protection' is too generic — it could refer to electrical protection class "
         "(Class I/II), short-circuit protection, or IP rating. "
         "'Degree of protection (IP code)' is the exact IEC 60529 terminology, ensuring "
         "unambiguous mapping to the IP XX rating system. "
         "Text type is retained; the regex ^(IP\\d\\d[A-Z]?)$ should also be applied "
         "to enforce valid IP code format. PIM: correct naming enables faceted IP-code "
         "filtering (e.g., 'IP65 or better') in product configurators."),

    26: ("See Row 20. 'Body Material' is a synonym for Housing Material and is merged "
         "into the canonical 'Housing material' attribute to eliminate redundancy. "
         "Text type is unchanged."),

    27: ("Vibration Resistance specifies the maximum vibration the switch can withstand "
         "per IEC 60068-2-6 (sinusoidal vibration test). Typically expressed as 'X g / "
         "Y Hz–Z Hz'. Name and type are consistent. No change required. "
         "PIM: critical for specifying switches in high-vibration environments."),

    28: ("Accessory Type classifies compatible add-on accessories "
         "(e.g., contact blocks, plunger heads). "
         "Name and type match the schema. No change required."),

    29: ("Contact Material specifies the contact alloy (e.g., silver cadmium oxide AgCdO, "
         "silver nickel AgNi, gold-plated). Contact material directly affects contact "
         "resistance, arc erosion, and minimum switching current. "
         "Name and type are consistent with the schema. No change required."),

    30: ("The 'switching principle' describes the fundamental electromechanical mechanism: "
         "slow-make/slow-break, snap-action, positive opening, or twin-contact. "
         "IEC 60947-5-1 uses 'switching principle' as the formal term. "
         "'Operation' is too vague and conflicts with other attributes describing "
         "actuator action. Rename aligns terminology with the standard. "
         "PIM: enables filtering by switching mechanism for safety-critical applications."),

    31: ("Supply Voltage is the operating power supply voltage for electronic (solid-state) "
         "limit switches. Text type prevents numeric filtering and range queries. "
         "Float enables voltage-range filtering (e.g., '10–30 V DC'). "
         "IEC 60947-5-1 specifies supply voltage as a numeric value. "
         "PIM: Float type is required for power supply compatibility checking in system design tools."),

    32: ("Supply Voltage UOM stores the voltage unit (V). Text type is appropriate "
         "as UOM strings vary (V AC, V DC). No change required."),

    33: ("See Row 14. 'Temperature Rating' is a non-standard name for the operating "
         "temperature range. Merging into 'Ambient operating temperature range' per "
         "IEC 60947-5-1 cl. 7.2.2 eliminates redundant attributes. "
         "Text type stores the range notation (e.g., '-40...+85°C')."),

    34: ("Temperature Rating UOM stores the temperature unit (°C or °F). "
         "Text type is appropriate. No change required."),

    35: ("'Contact Form', 'Contact Style' (Row 112), 'Configuration' (Row 119), and "
         "'Contact Configuration' (Row 94) all describe the NC/NO/CO wiring configuration. "
         "IEC 60947-5-1 uses 'contact configuration' as the standard term "
         "(e.g., 1NO, 1NC, 1NO+1NC). Merging eliminates duplicate columns. "
         "PIM: single canonical attribute enables reliable circuit-type filtering "
         "in wiring design and panel layout tools."),

    36: ("Maximum Pressure Measured is the rated pressure measurement range for "
         "pressure-sensitive limit switch variants. Constant type prevents numeric queries. "
         "Float type enables pressure-range filtering and unit conversion. "
         "PIM: numeric type is required for pressure-rated sensor selection in process control."),

    37: ("Maximum Pressure Measured UOM stores the pressure unit (e.g., bar, PSI). "
         "Constant type is appropriate (controlled vocabulary). No change required."),

    38: ("Operating Pressure specifies the line or actuating pressure required for "
         "pressure-operated switch variants. Constant type prevents numeric filtering. "
         "Float enables pressure setpoint comparison and engineering calculations. "
         "PIM: Float is required for pressure-range matching in hydraulic/pneumatic BOM tools."),

    39: ("Operating Pressure UOM stores the pressure unit. "
         "Constant type is appropriate. No change required."),

    40: ("'Dimensions' as a single text field (e.g., '80 x 40 x 30 mm') bundles multiple "
         "physical dimensions into one unstructured value that cannot be filtered or sorted "
         "numerically. Splitting into discrete Float fields (length, width, height) per "
         "IEC 60947-5-1 installation drawing convention enables dimensional compatibility "
         "checks. Float type is required for numeric dimension values. "
         "PIM: split dimensions enable 3D envelope filtering in panel layout tools and "
         "CAD/BIM integration workflows."),

    41: ("The companion UOM for the dimensions attribute was stored as Float — a data entry "
         "error since unit strings ('mm', 'in') are not numeric. Correcting to Constant "
         "(controlled vocabulary) aligns with the UOM pattern used throughout this schema. "
         "The attribute name update from 'Dimensions UOM' to 'Overall Dimensions UOM' "
         "maintains consistency with the target attribute 'Overall dimensions'. "
         "PIM: Constant UOM prevents accidental numeric storage of unit labels."),

    42: ("'For Use With' lists compatible series or families of accessories. "
         "Text type is appropriate for cross-reference strings. No change required."),

    43: ("Output Type describes the electrical output signal type "
         "(e.g., PNP, NPN, push-pull, relay). "
         "Name and type match the schema. No change required."),

    44: ("Action describes the actuator mechanical action style "
         "(e.g., spring-return, maintained). "
         "Text type is appropriate. No change required."),

    45: ("Wire size specifies the acceptable conductor cross-section range for termination "
         "(e.g., 0.5–2.5 mm²). 'Wire Size' is ambiguous — range implies a spread of "
         "acceptable values, not a single constant. 'Wire size range' explicitly identifies "
         "this as a range. Changing from Constant to Float enables numeric range queries "
         "and compatibility filtering with cable cross-sections. "
         "IEC 60947-5-1 specifies termination conductor cross-section range in mm². "
         "PIM: Float type and 'range' in the name prevent misuse as a single-point value."),

    46: ("The companion UOM for wire size range is renamed to 'Wire Size Range UOM' "
         "for consistency with Row 45. Constant type is retained (unit fixed to 'mm²' or 'AWG'). "
         "PIM: paired rename ensures attribute name alignment between value and UOM columns."),

    47: ("'Lever' is an ambiguous, product-specific label used in some catalogs to denote "
         "lever-style actuators. The canonical ETIM EC000024 attribute is 'Actuator Type', "
         "which classifies all actuator styles (pin plunger, roller lever, rod, cat whisker) "
         "under a single controlled vocabulary. Renaming eliminates a duplicated attribute "
         "alongside Row 8 ('Actuator Type') that already exists in the schema. "
         "Changing from Constant to Text provides flexibility for multi-value descriptions. "
         "PIM: merging lever variants under the canonical actuator type attribute prevents "
         "search fragmentation across product families."),

    48: ("Pitch specifies the terminal contact spacing (e.g., 3.5 mm, 5.08 mm) relevant "
         "for terminal block and DIN rail connector compatibility. Constant type prevents "
         "numeric matching; Float enables range queries and direct comparison to connector "
         "pitch specifications in panel design. "
         "IEC 60947-5-1 and connector standards specify pitch as a Float dimension. "
         "PIM: Float type enables pitch-based cross-reference filtering in accessory selection."),

    49: ("Pitch UOM is correctly typed as Constant (fixed to 'mm'). No change required."),

    50: ("'Roller Dimension' bundles the roller geometry into an unstructured text value "
         "that cannot be filtered numerically. Splitting into 'Roller diameter' as a "
         "discrete Float attribute enables mechanical compatibility checks between the "
         "roller and the cam profile. IEC 60947-5-1 and ETIM EC000024 specify roller "
         "diameter as a distinct Float attribute. "
         "PIM: Float enables dimensional range filtering for roller/cam matching calculations."),

    51: ("Shock Resistance specifies the peak shock acceleration the switch can withstand "
         "per IEC 60068-2-27. Values are typically '30g, 11 ms' — a compound description "
         "requiring Text type. Constant type implies a single fixed enumerated value, "
         "which is too restrictive for the variety of shock specification formats. "
         "PIM: Text enables pattern-matching and display of g-force/duration shock specs."),

    52: ("Protection Class refers to the IEC 61140 electrical safety class "
         "(Class I: earthed, Class II: double insulated). This is distinct from IP rating. "
         "Name and type are consistent. No change required. "
         "PIM: enables filtering for double-insulated (Class II) products in "
         "safety-critical or ungrounded installations."),

    53: ("Cable Type describes the cable construction for pre-wired limit switch variants "
         "(e.g., PVC jacketed, oil-resistant PUR). "
         "Name and type match the schema. No change required."),

    54: ("NEMA Rating specifies the NEMA enclosure rating (e.g., NEMA 1, 4, 4X, 13) "
         "for North American environmental protection classification. "
         "Text type is appropriate. No change required. "
         "PIM: NEMA rating is used alongside IP rating for North American "
         "product compliance filtering."),

    55: ("Number of Contacts is a count of discrete contact elements in the switch "
         "(e.g., 2, 4). Count attributes must be Integer — not Constant — to support "
         "arithmetic operations (e.g., total contact capacity calculation) and "
         "numeric range filters. Integer type also aligns with ETIM EC000024 "
         "data type conventions for count attributes. "
         "PIM: Integer enables filtering by contact count in circuit design configurators."),

    56: ("Cable Length is the pre-wired lead length for cable-connected variants. "
         "Float type is already correct and unchanged. No change required."),

    57: ("Cable Length UOM is correctly stored as Text (unit may be 'm' or 'ft'). "
         "No change required."),

    58: ("See Row 20. 'Casing Material' is a synonym for Housing Material and is merged "
         "into the canonical 'Housing material' attribute. Text type is unchanged."),

    59: ("Approvals lists certification marks (UL, CE, cUL, CSA, etc.). "
         "Text type supports multi-value approval strings. No change required. "
         "PIM: keyword-searchable approval field is essential for regulatory compliance filtering."),

    60: ("IEC 60947-5-1 defines 'number of poles' as the count of independent current paths "
         "through the switch; ETIM EC000024 uses 'Number of poles / contact circuits'. "
         "Adding '/ contact circuits' disambiguates the meaning for multi-break contacts. "
         "The original Float type is incorrect for a count — Integer is required. "
         "Float-to-Integer correction also prevents fractional values (e.g., 1.5 poles). "
         "PIM: Integer type enables numeric circuit-count filtering in panel wiring tools."),

    61: ("Output stores the output signal description for electronic variants. "
         "Constant type and name match the schema. No change required."),

    62: ("'Size' is an ambiguous catch-all that could mean overall dimensions, body size "
         "code, or outline. 'Overall dimensions' is the precise ETIM EC000024 term. "
         "Changing from Constant to Text allows the compound dimension notation "
         "('L x W x H mm') to be stored. Where dimensions are split (Row 40), "
         "this attribute holds the consolidated string as a display fallback. "
         "PIM: explicit naming prevents size from being confused with mounting footprint "
         "or catalogue size codes."),

    63: ("The UOM companion 'Size UOM' is renamed to 'Overall dimensions UOM' "
         "for consistency with the recommended attribute name in Row 62. "
         "Constant type is retained (controlled unit vocabulary). "
         "PIM: consistent value–UOM naming is required by unit-aware PIM platforms."),

    64: ("Weight is the product net weight in the specified UOM. "
         "Float type is already correct. No change required."),

    65: ("'Weight UOM' is unchanged in both name and type (Text→Text). "
         "The previous Change Type 'Rename' was incorrect. "
         "UOM stores the weight unit (e.g., 'g', 'kg', 'oz'). No change required."),

    66: ("Family Name identifies the product line or series family for cross-sell "
         "and related-product navigation. "
         "Text type and name match the schema. No change required."),

    67: ("'Type', 'Primary Type' (Row 90), and 'Switch Type' (Row 103) all describe "
         "the limit switch subtype classification (e.g., roller lever, safety, "
         "cable-pull, inductive). IEC 60947-5-1 and ETIM EC000024 define a single "
         "'Product type' attribute for this classification. Merging under "
         "'Product type (limit switch subtype)' eliminates three competing source columns "
         "and ensures a single authoritative classification per SKU. "
         "PIM: product type is the primary browse hierarchy node; a single canonical "
         "attribute prevents misclassification and navigation failures."),

    68: ("Housing type describes the mechanical form factor of the switch body "
         "(e.g., compact, standard, heavy-duty, zinc die-cast). "
         "'Housing Style' is an internal label; 'Housing type / form factor' aligns "
         "with ETIM EC000024 attribute naming and clarifies the physical meaning. "
         "Text type is unchanged. PIM: enables form-factor filtering in dimensional "
         "selection tools and CAD catalog plugins."),

    69: ("Actuator Material specifies the material of the actuator mechanism "
         "(e.g., stainless steel, polyamide, thermoplastic). "
         "Constant type implies a fixed single-value controlled vocabulary, but "
         "material descriptions vary across suppliers and may include compound values. "
         "Text allows flexible storage of material descriptions. "
         "PIM: Text enables keyword-based material filtering for chemical "
         "resistance and temperature performance."),

    70: ("Head Type classifies the actuator head assembly style fitted to the switch body "
         "(e.g., cross-roller, adjustable plunger, standard roller). "
         "Name and type already match the ETIM EC000024 attribute. No change required. "
         "PIM: Head Type drives the top-level actuator family filter in product selectors."),

    71: ("Mechanical life (endurance) is the number of no-load operating cycles the switch "
         "is guaranteed to survive, per IEC 60947-5-1 clause 8.2. "
         "'Mechanical Life' is an informal abbreviation; 'Mechanical life (endurance)' "
         "matches the IEC and ETIM terminology and distinguishes it from electrical life. "
         "The original Text type prevents numeric comparison of cycle counts "
         "(typically 1,000,000–30,000,000). Float is needed for threshold filtering "
         "and life expectancy calculations. "
         "PIM: Float enables service-life-based filtering in predictive maintenance catalogs."),

    72: ("The companion UOM for mechanical life is renamed to align with Row 71. "
         "Unit stores 'cycles' or 'million operations'. "
         "Text type is appropriate for unit labels. No change required."),

    73: ("Switching Voltage is the maximum voltage present at the contact at the moment "
         "of switching — a critical parameter for arc energy and contact erosion. "
         "Constant type prevents numeric range queries. Float enables specification "
         "filtering (e.g., 'switching voltage <= 250 V AC'). "
         "IEC 60947-5-1 defines switching voltage as a numeric Float attribute. "
         "PIM: Float type is required for contact rating table construction."),

    74: ("Switching Voltage UOM is correctly typed as Constant (fixed to 'V'). "
         "No change required."),

    75: ("'Connection', 'Connection Type' (Row 80), 'Termination' (Row 79), and "
         "'Terminal Style' (Row 131) all describe the electrical interface method "
         "(e.g., screw terminal, spring clamp, M12 connector, flying leads). "
         "IEC 60947-5-1 and ETIM EC000024 use 'Connection type' as the canonical term. "
         "Merging eliminates redundancy and ensures consistent classification. "
         "PIM: single attribute enables reliable filtering by interface type "
         "in wiring accessory selection workflows."),

    76: ("Standards lists applicable certifications and test standards "
         "(e.g., IEC 60947-5-1, EN 60947-5-1, UL 508, CSA C22.2). "
         "Text type is appropriate. No change required. "
         "PIM: keyword-searchable standards field supports compliance and export filtering."),

    77: ("Maximum switching frequency is the highest permissible rate of switching cycles "
         "per unit time (e.g., 60/min, 1 Hz) without exceeding the thermal limit of the contacts. "
         "'Speed' is ambiguous (could mean cam speed, actuator velocity, or line speed). "
         "IEC 60947-5-1 uses 'maximum switching frequency'. Text→Float is required "
         "because the value is a numeric physical quantity enabling derating calculations "
         "and cycle-rate comparisons. "
         "PIM: Float enables filtering by switching speed for high-frequency automation applications."),

    78: ("The companion UOM for maximum switching frequency (e.g., '/min', 'Hz') is "
         "renamed consistently with Row 77. Text type is appropriate. No change required."),

    79: ("See Row 75. 'Termination' is merged into the canonical 'Terminal type' attribute "
         "alongside Connection (Row 75), Connection Type (Row 80), and Terminal Style (Row 131). "
         "Terminology standardisation eliminates duplicate attribute columns and ensures "
         "consistent wiring interface classification."),

    80: ("See Row 75. 'Connection Type' is merged into the canonical 'Connection type' "
         "(note consistent lowercase). The slight case difference was causing duplicate "
         "attribute entries in PIM systems. Merge consolidates both under a single record."),

    81: ("See Row 14. 'Temperature Operating Range' is one of six source attributes that "
         "all describe the same property (ambient operating temperature range). "
         "This row was incorrectly labelled 'Attribute Split' — the source column is not "
         "being split; rather, it is being merged with other temperature range attributes "
         "into the single canonical target. Change Type corrected to 'Attribute Merge'."),

    82: ("The companion UOM for Temperature Operating Range is renamed to align with the "
         "merged canonical target 'Ambient operating temperature range'. "
         "Text type is appropriate (unit is °C or °F). No change required."),

    83: ("'Contact Type' is ambiguous — it could mean contact material, contact form, or "
         "contact actuation mechanism. 'Contact operation type' precisely describes the "
         "actuation principle: slow-action, snap-action, or positive-opening. "
         "This distinction is safety-critical: IEC 60947-5-1 Annex K mandates positive-opening "
         "contacts for safety-related applications. "
         "Constant→Text allows compound descriptions (e.g., 'Snap-action, positive opening'). "
         "PIM: unambiguous naming prevents contact operation type from being confused "
         "with contact configuration (NO/NC/CO)."),

    84: ("Power is the rated electrical power dissipated in the switch contacts at rated load "
         "(typically VA for AC, W for DC). Constant type prevents numeric power calculations "
         "and range filtering. Float enables comparison and thermal budget calculations. "
         "IEC 60947-5-1 specifies rated power as a numeric Float value. "
         "PIM: Float enables thermal load calculation when sizing enclosures and cooling."),

    85: ("Power UOM stores the unit (VA or W). Constant type is appropriate. No change required."),

    86: ("Actuator Length is the physical length of the actuator arm or plunger, "
         "relevant for clearance and cam geometry. Float type is already correct. "
         "No change required."),

    87: ("This row is a duplicate of Row 86 with a different source data type (Constant). "
         "This likely reflects different product families where actuator length was stored "
         "as a controlled vocabulary rather than a float. Both converge to the same "
         "canonical attribute 'Actuator Length'. No change required."),

    88: ("Accommodation Terminal specifies the maximum conductor cross-section that "
         "the terminal can accommodate. Constant type and name match the schema. "
         "No change required."),

    89: ("Accommodation Terminal UOM stores the cross-section unit (mm² or AWG). "
         "Constant type is appropriate. No change required."),

    90: ("See Row 67. 'Primary Type' is one of three source attributes merged into "
         "'Product type (limit switch subtype)'. Merging eliminates classification "
         "fragmentation and ensures one authoritative type per SKU."),

    91: ("Number of Pins specifies the contact pin count for connector-style "
         "limit switches (e.g., M12 4-pin). Pin count is a discrete quantity requiring "
         "Integer type — Constant implies a controlled vocabulary, preventing arithmetic "
         "operations and range filtering. Integer aligns with ETIM EC000024 conventions. "
         "PIM: Integer enables pin-count filtering for connector compatibility matching."),

    92: ("Contact ratings describe the electrical load the contacts can switch "
         "(e.g., AC-15: 10A at 230V). 'Contact Rating' conflates multiple rating "
         "parameters into one column. 'Contact ratings (rating table)' signals that "
         "the attribute stores a table reference or structured multi-value entry "
         "covering multiple utilisation categories per IEC 60947-5-1. "
         "PIM: the table reference notation guides downstream ETL systems to expand "
         "this attribute into a proper rating matrix."),

    93: ("The companion UOM for contact ratings is renamed consistently with Row 92. "
         "Constant type is retained. PIM: paired renaming maintains value–UOM alignment."),

    94: ("'Contact Configuration' (with capital C) is merged into 'Contact configuration' "
         "(lowercase) to resolve the case inconsistency between source attribute names. "
         "This is one of four attributes (Rows 35, 94, 112, 119) describing the same "
         "NC/NO/CO configuration property. Merging ensures a single canonical attribute."),

    95: ("Number of Conductors specifies how many conductors are in the cable "
         "(relevant for pre-wired variants). Count attributes require Integer — Constant "
         "prevents arithmetic and numeric range filtering. Integer aligns with ETIM EC000024. "
         "PIM: Integer enables conductor-count filtering for multi-core cable selection."),

    96: ("'Type Port' is unchanged in both name and type (Text→Text). "
         "The previous Change Type 'Rename' was incorrect. No change required. "
         "PIM: this attribute describes the connector port type (e.g., M12, M8, M23)."),

    97: ("'Base Material' is unchanged in name but changes in type from Constant to Text. "
         "The previous Change Type 'Rename' was incorrect (name is the same). "
         "Constant implies a limited controlled vocabulary; Text allows for material "
         "descriptions that vary across product families and suppliers. "
         "PIM: Text enables flexible material keyword search across mixed product catalogs."),

    98: ("'Current Ratings', 'Current Rating' (Row 122), and the ETIM canonical attribute "
         "is 'Rated operational current (Ie)' per IEC 60947-5-1. Ie is the current at "
         "which the contact can reliably make and break under the utilisation category "
         "conditions. The IEC subscript notation (Ie) prevents confusion with "
         "conventional thermal current (Ith) or rated current (In). "
         "Text→Float allows numeric current values for range queries. "
         "PIM: Ie is a mandatory attribute for ETIM classification and "
         "contact rating table population."),

    99: ("The companion UOM for rated operational current is renamed to align with Row 98. "
         "Constant type is retained (unit is 'A'). No change required."),

    100: ("Spring Material specifies the contact spring alloy "
          "(e.g., stainless steel, beryllium copper). Affects spring force consistency "
          "and fatigue life. Name and type match the schema. No change required."),

    101: ("Application describes the intended use context "
          "(e.g., conveyor, safety door, cam-operated). "
          "Text type supports free-form application descriptions. No change required."),

    102: ("Color specifies the switch body or actuator colour for identification. "
          "Text type is appropriate. No change required."),

    103: ("See Row 67. 'Switch Type' is one of three source attributes merged into "
          "'Product type (limit switch subtype)'. Merging prevents three separate "
          "type classification columns from holding conflicting values."),

    104: ("'Seal Type' is unchanged in name but changes in type from Constant to Boolean. "
          "The previous Change Type 'Rename' was incorrect (name is the same). "
          "IMPORTANT DATA TYPE NOTE: Changing 'Seal Type' to Boolean is semantically "
          "questionable — 'Seal Type' implies a descriptive type (e.g., 'IP67 O-ring', "
          "'double lip seal', 'none') rather than a true/false flag. "
          "If the intent is 'Is the switch sealed?', the attribute should be renamed to "
          "'Is Sealed' or 'Sealed (Y/N)' with Boolean type. "
          "If it describes the seal mechanism, the recommended type should be "
          "Text or Constant, not Boolean. Review required before implementation."),

    105: ("Repeat accuracy is the deviation between successive switching points at the "
          "same mechanical approach speed and direction, per IEC 60947-5-1. It is a key "
          "precision indicator for cam-operated machine tools and robotics. "
          "'Accuracy' is non-specific; 'Repeat accuracy' matches the ETIM EC000024 "
          "attribute name and IEC 60947-5-1 clause 5.5.3 definition. "
          "Text→Float is required: accuracy is a numeric tolerance (e.g., ±0.5 mm) "
          "that must support comparison and filtering. "
          "PIM: Float enables precision-based product selection for high-accuracy positioning applications."),

    106: ("The companion UOM for repeat accuracy (typically 'mm') is renamed "
          "consistently with Row 105. Constant type is retained. No change required."),

    107: ("Humidity range specifies the maximum non-condensing relative humidity for "
          "reliable switch operation (e.g., '35–85% RH'). "
          "'Humidity' alone is ambiguous. 'Humidity range (non-condensing)' explicitly "
          "states the operating condition and excludes condensation, consistent with "
          "IEC 60068-2-78 test conditions. Constant→Text allows range notation storage. "
          "PIM: explicit non-condensing qualifier prevents misapplication in "
          "high-humidity outdoor environments."),

    108: ("The companion UOM for humidity range (typically '% RH') is renamed "
          "consistently with Row 107. Constant type is retained. No change required."),

    109: ("Jacket Color specifies the insulation jacket colour for pre-wired cable variants. "
          "Constant type and name match the schema. No change required."),

    110: ("Cord Length is the cable lead length for pre-wired variants. "
          "Float type is already correct. No change required."),

    111: ("Cord Length UOM stores the length unit (m or ft). "
          "Constant type is appropriate. No change required."),

    112: ("See Row 35. 'Contact Style' is a synonym for contact configuration "
          "(NC/NO/CO) and is merged into the canonical 'Contact configuration' attribute. "
          "Text type is unchanged."),

    113: ("Operating force (OF) is the minimum force required to actuate the switch "
          "from the free position to the operating position, per IEC 60947-5-1 cl. 5.5.2. "
          "The IEC subscript notation OF in the recommended name prevents confusion with "
          "releasing force (RF) or pretravel force. "
          "Text→Float is required: OF is a numeric force value (e.g., 5 N, 12 N) "
          "that must support range queries and mechanical engineering calculations. "
          "PIM: Float enables force-based filtering for applications with limited "
          "actuating force (e.g., lightweight robotics end-effectors)."),

    114: ("The companion UOM for operating force (typically 'N') is renamed "
          "consistently with Row 113. Text type is appropriate. No change required."),

    115: ("WARRANTY DATA TYPE NOTE: Warranty is stored as Text (e.g., '2 years', "
          "'Limited Lifetime') but the recommended type is Float. "
          "Changing to Float requires all warranty values to become numeric, "
          "losing 'Lifetime', 'Limited', and 'Conditional' qualifiers. "
          "For warranty duration in months/years, Float is appropriate only if "
          "a companion UOM attribute (e.g., 'Warranty UOM': months/years) is also created "
          "and non-numeric warranty types are handled separately. "
          "Recommended: add 'Warranty UOM' companion attribute and map special values "
          "to a 'Warranty Type' Constant attribute. Review before applying Float conversion."),

    116: ("Series identifies the product line family within the manufacturer's portfolio. "
          "Text type is appropriate. No change required."),

    117: ("Frequency Range specifies the operating frequency range for "
          "electronic/inductive variant supply. "
          "Constant type and name match the schema. No change required."),

    118: ("Frequency Range UOM stores the frequency unit (Hz). "
          "Constant type is appropriate. No change required."),

    119: ("See Row 35. 'Configuration' is a generic term used for contact configuration "
          "(NC/NO/CO) and is merged into the canonical 'Contact configuration' attribute. "
          "Merging eliminates the ambiguity of 'Configuration' which could apply to "
          "any switch parameter."),

    120: ("NEMA-rated DC voltage follows the same logic as Row 6 for AC. "
          "'DC Voltage' is renamed to 'NEMA DC voltage rating' per NEMA ICS 14 standards "
          "to distinguish from IEC DC operational voltage (Ue). "
          "Constant→Float is required: voltage is a numeric value requiring "
          "range queries and derating calculations. "
          "PIM: Float enables voltage-range filtering in North American DC drive applications."),

    121: ("The companion UOM for NEMA DC voltage rating is renamed consistently with Row 120. "
          "Constant type is retained (unit is 'V'). No change required."),

    122: ("See Row 98. 'Current Rating' (singular) is a variant of 'Current Ratings' "
          "and both are merged into 'Rated operational current (Ie)'. "
          "Text→Float enables current-value range filtering."),

    123: ("The companion UOM for Ie is renamed consistently with Row 122. "
          "Text type is retained. No change required."),

    124: ("See Row 14. 'Maximum Operating Temperature' is one of six source attributes "
          "all describing the ambient operating temperature range. "
          "Change Type corrected from 'Rename' to 'Attribute Merge' because the same target "
          "'Ambient operating temperature range' is shared with Rows 14, 33, 81, 150, and 155. "
          "The maximum temperature value should ideally be captured as a separate "
          "'Maximum ambient temperature' Float attribute, but merging into the range "
          "notation is accepted for interim harmonisation. "
          "PIM: merge ensures a single queryable temperature spec per SKU."),

    125: ("The companion UOM for Maximum Operating Temperature is renamed to align "
          "with the merged canonical attribute. "
          "Text type is appropriate (unit is °C or °F). No change required."),

    126: ("Body Style describes the switch housing form (e.g., compact, standard, metal). "
          "Text type and name match the schema. No change required."),

    127: ("Frequency here refers to the rated AC supply frequency (e.g., 50 Hz, 60 Hz, "
          "50/60 Hz). Constant type implies a fixed controlled vocabulary, but Integer "
          "is more appropriate for a discrete numeric frequency value that may need "
          "arithmetic comparison (e.g., filter by 60 Hz for North American market). "
          "ETIM EC000024 recommends Integer for frequency. "
          "PIM: Integer enables filtering by supply frequency for market-specific catalogs."),

    128: ("Frequency UOM stores the unit ('Hz'). Constant type is appropriate. No change required."),

    129: ("Overall Length is a physical dimension attribute that must be stored as Float "
          "to support numeric comparison, unit conversion, and dimensional compatibility checks. "
          "Constant type prevents these operations. "
          "IEC 60947-5-1 installation dimensions are Float values. "
          "PIM: Float is mandatory for any dimension used in 3D clearance verification."),

    130: ("Overall Length UOM stores the length unit ('mm' or 'in'). "
          "Constant type is appropriate. No change required."),

    131: ("See Row 75. 'Terminal Style' is one of four attributes describing the "
          "electrical interface method and is merged into the canonical 'Terminal type'. "
          "Merging ensures consistent wiring interface classification across product families."),

    132: ("'Connector Type' describes the mechanical mating interface for "
          "plug-in connection (e.g., M12 A-code, M8, 7/8 in). "
          "'Connector interface' is a more precise term that encompasses both the "
          "mechanical standard and the pinout configuration. "
          "IEC 61076-2-101 (M12 connectors) uses 'interface' terminology. "
          "PIM: enables connector-type filtering for pre-wired and quick-disconnect variants."),

    133: ("Dielectric withstand voltage is the test voltage (1500 V AC for 1 minute) "
          "applied between circuits to verify insulation integrity per IEC 60947-5-1 cl. 8.3.3. "
          "'Dielectric Strength' is an informal term that conflates material property "
          "with the test voltage value. 'Dielectric withstand voltage' matches the IEC term "
          "and clarifies it is a test parameter, not a material characteristic. "
          "Float type is already correct. PIM: enables filtering for high-voltage insulation requirements."),

    134: ("The companion UOM for dielectric withstand voltage (kV or V) is renamed "
          "consistently with Row 133. Constant type is retained. No change required."),

    135: ("'Mounting Type' is too generic — it could apply to DIN rail, panel, "
          "or surface mounting. 'Mounting style' is the ETIM EC000024 attribute name "
          "and describes the physical mounting method (e.g., panel mount, DIN rail, "
          "surface mount). Rename clarifies the intended scope. "
          "PIM: enables mounting-method filtering in panel layout and DIN rail configurators."),

    136: ("Rated operational voltage (Ue) is the IEC 60947 designation for the maximum "
          "voltage at which the switch is rated to operate continuously. "
          "IEC 60947-5-1 uses Ue to distinguish from insulation voltage (Ui) and "
          "impulse voltage (Uimp). 'Voltage Rating' is non-standard and conflates "
          "these three distinct voltage parameters. "
          "Text→Float is required for numeric voltage filtering, range queries, and "
          "derating calculations. PIM: Ue is a mandatory attribute for IEC product classification; "
          "Float enables voltage-tier filtering (e.g., 230 V, 400 V, 690 V)."),

    137: ("The companion UOM for Ue (typically 'V') is renamed consistently with Row 136. "
          "Text type is appropriate. No change required."),

    138: ("Case Material specifies the outer case/shell material, potentially distinct "
          "from the housing body material (Row 20). Text type is appropriate. "
          "No change required. Note: if 'Case Material' and 'Housing material' refer to "
          "the same physical component, consider merging into the canonical housing attribute."),

    139: ("Protocol Type specifies the fieldbus or IO-Link protocol version. "
          "Constant type (controlled vocabulary) and name match the schema. No change required."),

    140: ("Number of Levels specifies the number of switching levels or detent positions "
          "(relevant for multi-position switches). Count attributes require Integer — "
          "Constant prevents arithmetic and range filtering. "
          "PIM: Integer enables level-count filtering in multi-step cam applications."),

    141: ("Releasing force (RF) is the force at which the actuator returns from the "
          "operating position to the free position, per IEC 60947-5-1 cl. 5.5.2. "
          "'Release Force' is the informal version; 'Releasing force (RF)' uses the "
          "IEC subscript notation and distinguishes it from operating force (OF). "
          "Text→Float is required for numeric force comparison and spring-return calculations. "
          "PIM: Float enables RF/OF ratio filtering for checking snap-action characteristics."),

    142: ("Enclosure Rating describes the IP or NEMA environmental protection rating "
          "of the overall enclosure assembly. "
          "Text type and name match the schema. No change required."),

    143: ("Number of Decks specifies the number of contact block layers in stackable "
          "switch configurations. Count attributes require Integer — Constant prevents "
          "arithmetic and range filtering. "
          "PIM: Integer enables deck-count filtering in modular panel assembly configurators."),

    144: ("Number of Positions specifies the detent position count for multi-position "
          "rotary or selector switches. Float→Integer is the correct correction: "
          "position count is always a whole number and Float allows fractional values "
          "(e.g., 2.5 positions) that are physically meaningless. "
          "PIM: Integer prevents decimal position counts and enables exact numeric filtering."),

    145: ("See Row 20. 'Housing Material' (capital H and M) is merged into the "
          "canonical 'Housing material' (lowercase) to resolve case inconsistency "
          "between duplicate source columns. Merge ensures one authoritative housing "
          "material value per SKU."),

    146: ("Range Input specifies the input range for sensing variants. "
          "Constant type and name match the schema. No change required."),

    147: ("Input Voltage is the supply voltage for powered (electronic) switch variants. "
          "Text type prevents numeric voltage filtering. Float enables range queries "
          "and power supply compatibility checks. "
          "IEC 60947-5-1 specifies supply voltage as a numeric value. "
          "PIM: Float enables input voltage filtering for DC supply compatibility in "
          "I/O module and sensor integration."),

    148: ("Actuator action / approach direction specifies the direction from which the "
          "actuating force must approach (e.g., top, side, omnidirectional) and the "
          "mechanical action style (positive, rotary, linear). "
          "'Actuation' alone is too generic and does not communicate the direction component. "
          "ETIM EC000024 separates actuator type (Row 8) from approach direction as "
          "distinct attributes for precise mechanical specification. "
          "PIM: enables directional filtering for cam-approach constrained applications."),

    149: ("Function describes the switch's electrical function "
          "(e.g., safety, monitoring, control). "
          "Constant type and name match the schema. No change required."),

    150: ("See Row 14. 'Operating Temperature' is one of six source attributes merged "
          "into 'Ambient operating temperature range' per IEC 60947-5-1. "
          "Merge consolidates all temperature range data under one canonical attribute."),

    151: ("The companion UOM for Operating Temperature is renamed consistently with "
          "the merged target. Text type is appropriate. No change required."),

    152: ("Duplicate 'Series' row. Name and type match the schema. "
          "Review for data deduplication — this is likely a duplicate entry from "
          "multiple source product families. No change required to schema definition."),

    153: ("Hole Size specifies the mounting hole or clearance diameter for installation. "
          "Text type prevents numeric comparison. Float enables dimensional filtering "
          "and direct matching against panel knockout sizes. "
          "IEC 60947-5-1 installation dimensions are Float values. "
          "PIM: Float is required for hole-size compatibility filtering in panel design tools."),

    154: ("Hole Size UOM stores the hole dimension unit (mm or in). "
          "Constant type is appropriate. No change required."),

    155: ("See Row 14. 'Minimum Operating Temperature' is one of six source attributes "
          "merged into 'Ambient operating temperature range'. "
          "Change Type corrected from 'Rename' to 'Attribute Merge' for consistency "
          "with the other temperature range merge rows. "
          "The minimum temperature value ideally belongs in a 'Minimum ambient temperature' "
          "Float attribute for individual lower-limit filtering, but the range merge "
          "is accepted for interim harmonisation."),

    156: ("The companion UOM for Minimum Operating Temperature is renamed consistently "
          "with the merged target. Text type is appropriate. No change required."),

    157: ("Dimensions Nested specifies the overall dimensions when the product is "
          "in its nested or closed configuration (relevant for hinged or adjustable "
          "actuator heads). Float type is already correct. No change required."),

    158: ("Leaf Node is a metadata classification flag used in product taxonomy trees. "
          "Constant type is appropriate for a controlled hierarchy marker. No change required."),

    # ---- New Attributes (Missing) — rows 159–191 ----
    159: ("Number of NO (normally open) contacts is a mandatory circuit specification "
          "per IEC 60947-5-1. Engineers must know how many contacts open when the "
          "switch is actuated to design the control circuit correctly. "
          "Without this attribute, panel designers cannot verify whether the switch "
          "provides enough NO circuits for their wiring diagram. "
          "Integer type is required. ETIM EC000024 includes this as a discrete count attribute. "
          "PIM: enables contact-circuit-type filtering alongside NC count for wiring BOM generation."),

    160: ("Number of NC (normally closed) contacts is equally mandatory — NC contacts "
          "are used in safety interlocks where the circuit must be open in the "
          "non-actuated state. Without this attribute, safe-state circuit design "
          "cannot be verified from catalog data alone. "
          "Integer type required. IEC 60947-5-1 and ETIM EC000024 both require NC count. "
          "PIM: enables NC/NO ratio filtering for safety interlock specification."),

    161: ("Positive opening operation is a binary safety attribute defined in "
          "IEC 60947-5-1 Annex K and EN ISO 13850. A positively driven (direct-opening) NC "
          "contact is mechanically linked to the actuator so it CANNOT remain closed "
          "if the actuator is fully depressed — preventing stuck-contact failures in "
          "safety circuits. This is mandatory for EN ISO 13849 safety function category "
          "B and above. Boolean (True = positive opening guaranteed) is the correct type. "
          "Without this attribute, safety engineers cannot identify compliant products "
          "from catalog data. PIM: highest-priority missing attribute for safety-rated applications."),

    162: ("Utilisation category classifies the electrical load type the switch is rated for "
          "per IEC 60947-5-1: AC-15 (inductive AC coils), DC-13 (inductive DC coils), "
          "AC-12 (resistive AC), DC-12 (resistive DC). "
          "The category determines the applicable make/break current ratings and contact life. "
          "Without utilisation category, specifiers cannot verify the switch is rated "
          "for the actual load (e.g., motor contactor coils vs. PLC inputs). "
          "Text type stores the IEC category code. "
          "PIM: mandatory for contact rating table construction and compliance documentation."),

    163: ("Conventional enclosed thermal current (Ith) is the maximum continuous current "
          "through the enclosed switch without exceeding the temperature limits of the "
          "insulation, per IEC 60947-5-1 cl. 4.4.1. "
          "Ith is distinct from rated operational current (Ie) and is used for "
          "thermal calculation of enclosure temperature rise. "
          "Float type required. Missing this attribute prevents proper thermal budget "
          "calculations for enclosed panel designs. "
          "PIM: required for ETIM EC000024 completeness and thermal compliance filtering."),

    164: ("Rated insulation voltage (Ui) is the maximum voltage to which insulation is "
          "coordinated, per IEC 60947-5-1 and IEC 60664. Ui >= Ue and sets the "
          "dielectric clearance requirements. It is distinct from operational voltage (Ue). "
          "Float type required. Without Ui, insulation coordination per IEC 60664 "
          "cannot be verified for the installation environment. "
          "PIM: required for high-voltage system specification and safety documentation."),

    165: ("Rated impulse withstand voltage (Uimp) is the peak surge/lightning voltage "
          "the insulation must withstand without flashover, per IEC 60947-5-1 and "
          "IEC 60664-1. Uimp determines the overvoltage category (I–IV) and clearance "
          "distances. Float type required (values in kV). "
          "Without Uimp, surge immunity and clearance compliance cannot be verified. "
          "PIM: required for industrial environments with significant surge risk (OVC III/IV)."),

    166: ("NEMA AC make current rating is the maximum current the contacts can close "
          "onto under NEMA ICS 14 conditions. This is distinct from the IEC Ie rating "
          "because NEMA tests use different load power factors. "
          "Float type required. Without this attribute, North American specifiers cannot "
          "verify NEMA-rated make performance for motor starter applications. "
          "PIM: required for UL/NEMA listed products distributed in North American market."),

    167: ("NEMA AC break current rating is the maximum current the contacts can interrupt "
          "under NEMA ICS 14 test conditions (typically at 0.45 power factor). "
          "Float type required. Break current rating is the critical limiting parameter "
          "for contact selection in inductive load switching. "
          "PIM: paired with make rating (Row 166) for complete NEMA contact rating documentation."),

    168: ("NEMA DC make current rating is the maximum DC current the contacts can close "
          "onto per NEMA ICS 14. DC breaking is significantly harder than AC due to the "
          "absence of a natural current zero; make and break ratings are both critical. "
          "Float type required. PIM: required for North American DC drive and solar applications."),

    169: ("NEMA DC break current rating is the maximum DC current the contacts can interrupt "
          "per NEMA ICS 14. DC arc interruption requires adequate contact gap and "
          "arc quench design. Float type required. "
          "PIM: high-priority missing attribute for DC control circuit applications."),

    170: ("Minimum switching voltage is the lowest voltage at which reliable contact "
          "switching is guaranteed without contact 'stiction' (adhesion at micro-loads). "
          "Below this value, the oxide layer on contact surfaces may prevent clean "
          "make/break. Float type required (typically 5–24 V DC). "
          "Missing this attribute prevents specifiers from identifying switches suitable "
          "for low-voltage logic signal switching. "
          "PIM: critical for PLC input and NAMUR sensor compatibility checking."),

    171: ("Minimum switching current is the lowest current that will reliably commutate "
          "through the contact without 'dry circuit' failures caused by contact film "
          "formation. Float type required (typically 1–100 mA). "
          "Without this attribute, switches cannot be reliably specified for "
          "low-power logic circuits. IEC 60947-5-1 cl. 5.4 covers micro-load ratings. "
          "PIM: mandatory for NAMUR/sensor interface specification and micro-load applications."),

    172: ("Contact resistance is the DC resistance across a closed contact pair "
          "(typically 5–50 mΩ), measured per IEC 60947-5-1. "
          "High contact resistance causes voltage drop and heat generation under load. "
          "Float type required (in mΩ). Without this attribute, voltage drop calculations "
          "for 24 V DC control circuits cannot be completed. "
          "PIM: required for precision circuit design and contact quality grade filtering."),

    173: ("Insulation resistance is the resistance between open contact poles or between "
          "contacts and earth (typically >100 MΩ at 500 V DC per IEC 60947-5-1). "
          "Low insulation resistance indicates moisture ingress or insulation degradation. "
          "Float type required (in MΩ). Missing this attribute prevents leakage current "
          "calculations for safety-isolated circuits. "
          "PIM: required for high-impedance circuit and medical-grade product filtering."),

    174: ("Actuator head adjustability indicates whether the actuator head can be "
          "repositioned on the switch body (rotated or swivelled) to suit different "
          "cam approach directions. Boolean (True = adjustable) is the correct type. "
          "Without this attribute, engineers cannot identify which switches allow "
          "head repositioning without ordering a different variant. "
          "IEC 60947-5-1 and ETIM EC000024 include this as an installation feature flag. "
          "PIM: Boolean enables filtering for head-adjustable switches in flexible machine cell designs."),

    175: ("Operating position (OP) is the actuator position (in mm or degrees from free "
          "position) at which the switch changes state, per IEC 60947-5-1 cl. 5.5.1. "
          "Float type required. OP is critical for cam geometry design to ensure "
          "the switch operates within the cam lift profile. "
          "PIM: Float enables OP-based cam geometry verification in motion design tools."),

    176: ("Free position (FP) is the actuator rest position with no applied force, "
          "measured from the mechanical reference datum, per IEC 60947-5-1. "
          "Float type required. FP, together with OP and OT, defines the complete "
          "travel envelope of the actuator. "
          "PIM: required for travel-clearance verification in machine guarding applications."),

    177: ("Pre-travel (PT) is the distance the actuator travels from the free position "
          "to the operating position, per IEC 60947-5-1 cl. 5.5.1. "
          "Float type required (in mm or degrees). "
          "PT determines how close the cam must approach before the switch actuates; "
          "too large a PT causes late switching; too small risks damage. "
          "PIM: Float enables pre-travel-based cam clearance calculation and "
          "machine safety distance verification per EN ISO 13855."),

    178: ("Overtravel (OT) is the additional travel permitted beyond the operating "
          "position without damaging the switch, per IEC 60947-5-1 cl. 5.5.1. "
          "Float type required (in mm or degrees). "
          "Insufficient overtravel accommodation risks mechanical damage to the actuator "
          "under cam overshoot. OT is a critical mechanical robustness parameter. "
          "PIM: Float enables OT-based safety margin calculation in cam drive design."),

    179: ("Total travel (TT) is the sum of pre-travel and overtravel (TT = PT + OT), "
          "representing the full actuation stroke, per IEC 60947-5-1. "
          "Float type required. TT determines the minimum cam lift height required. "
          "PIM: TT is the most commonly used travel spec for initial cam height selection; "
          "Float enables direct comparison across product families."),

    180: ("Differential travel (DT) is the actuator travel between the operating position "
          "(make/break) and the release position (reset), per IEC 60947-5-1 cl. 5.5.1. "
          "Float type required (in mm or degrees). "
          "Small DT can cause contact chatter on vibrating machinery; "
          "large DT creates slow release response. DT is critical for "
          "bounce-free switching in high-cycle applications. "
          "PIM: Float enables DT-based selection for vibration-tolerant applications."),

    181: ("Electrical life (endurance) is the number of on-load switching operations "
          "the switch contacts can perform at rated voltage, current, and utilisation "
          "category before end-of-life, per IEC 60947-5-1 cl. 8.3. "
          "This is distinct from mechanical life (Row 71) which measures no-load cycles. "
          "Float type required (typically 100,000–1,000,000 cycles). "
          "Without electrical life, maintenance interval calculation for contact "
          "replacement is impossible. "
          "PIM: Float enables service life comparison and predictive maintenance scheduling."),

    182: ("Ambient storage temperature range specifies the temperature range for shipping "
          "and warehouse storage, per IEC 60068-1. It differs from operating range (Row 14) "
          "and is typically wider (e.g., -40°C to +85°C). "
          "Text type stores the range notation. Without this attribute, logistics and "
          "warehouse management systems cannot verify storage condition compliance. "
          "PIM: required for cold-chain logistics documentation and outdoor storage compliance."),

    183: ("Pollution degree classifies the contamination level of the switch's operating "
          "environment per IEC 60664-1: Degree 1 (no pollution), 2 (non-conductive "
          "pollution only), 3 (conductive pollution possible), 4 (continuous conductive "
          "pollution). Pollution degree determines required clearance distances and "
          "enclosure protection requirements. Text type stores the degree value. "
          "Without this attribute, insulation coordination per IEC 60664 cannot be "
          "completed. PIM: required for installation environment classification and "
          "ATEX/hazardous area documentation."),

    184: ("Altitude limit specifies the maximum installation altitude above sea level "
          "without voltage derating, per IEC 60947-1 cl. 8.1.2 (standard: 2000 m). "
          "Above this altitude, reduced air density lowers dielectric strength, "
          "requiring voltage derating or special insulation. Float type required (in m). "
          "PIM: required for high-altitude industrial installations (mining, mountain plants)."),

    185: ("Flammability rating (UL 94) specifies the plastic enclosure's flame resistance "
          "classification per UL 94 (V-0, V-1, V-2, HB). V-0 is required in "
          "IEC 60947 for parts near arc-generating contacts. Without this attribute, "
          "fire safety compliance for UL 508A panel builders cannot be verified. "
          "Text type stores the UL 94 classification code. "
          "PIM: mandatory for UL panel shop compliance filtering and NFPA 79 documentation."),

    186: ("Contact block system describes whether the switch body uses a modular "
          "contact block architecture allowing field-addition of contact blocks "
          "(e.g., Telemecanique OsiSense, Allen-Bradley 802T style). "
          "Text type stores the system type or compatibility group. "
          "Without this attribute, field engineers cannot identify which "
          "contact blocks are compatible with the base unit. "
          "PIM: enables modular system family filtering for accessories cross-sell."),

    187: ("Number of contact blocks specifies how many contact blocks are currently "
          "installed or the maximum stackable count. Float type (should be Integer — "
          "review recommended). Without this attribute, the contact count capacity "
          "of the modular switch cannot be determined from catalog data. "
          "PIM: enables filtering by total contact block capacity for modular panel builds."),

    188: ("Number of auxiliary contacts specifies the count of additional signalling "
          "or feedback contacts beyond the primary switching contacts. "
          "Auxiliary contacts are used for position feedback, PLC monitoring, or "
          "indicator circuit wiring. Float type (should be Integer — review recommended). "
          "PIM: enables auxiliary contact count filtering for feedback circuit design."),

    189: ("Switching element action type describes the snap mechanism: "
          "slow-make/slow-break (following actuator motion) or snap-action "
          "(pre-stressed spring mechanism that fires at a set point). "
          "Snap-action switches have defined operating force independent of approach speed. "
          "IEC 60947-5-1 cl. 4.3 classifies switching elements by action type. "
          "Text type. Without this attribute, slow-action vs. snap-action selection "
          "for vibration environments is not possible. "
          "PIM: required for noise-sensitive applications where snap switching is mandated."),

    190: ("Indicator / status display describes whether the switch has a visible "
          "position indicator, LED, or mechanical flag showing the switching state. "
          "This is important for maintenance personnel to visually verify switch status "
          "without energising the circuit. Text type stores the indicator type. "
          "PIM: enables filtering for switches with local visual feedback, "
          "important for safety monitoring and field diagnostics."),

    191: ("Manual override / test function indicates whether the switch has a "
          "built-in test button or lever allowing the output to be activated "
          "without physical actuator movement. "
          "This feature is used during commissioning, wiring verification, and "
          "functional safety testing (IEC 62061 / EN ISO 13849). "
          "Boolean (True = manual override present). "
          "PIM: enables filtering for test-capable switches in safety-rated systems "
          "requiring periodic proof-test capability."),
}

# ---------------------------------------------------------------------------
# Apply corrections and build output DataFrame
# ---------------------------------------------------------------------------
out_rows = []
for i, row in df.iterrows():
    orig    = g(row, 'Original Attribute Name')
    rec     = g(row, 'Recommended Attribute Name')
    ct      = g(row, 'Change Type')
    odt     = g(row, 'Original Data Type')
    rdt     = g(row, 'Recommended Data Type')
    orig_u  = g(row, 'Original Units')
    rec_u   = g(row, 'Recommended Units')
    regex   = g(row, 'Regex Pattern')
    old_exp = g(row, 'Explanation of Recommendation')

    corrected_ct = CT_OVERRIDES.get(i, ct)
    better_exp   = EXPLANATIONS.get(i, '')

    out_rows.append({
        'Original Attribute Name':    orig,
        'Recommended Attribute Name': rec,
        'Change Type':                corrected_ct,
        'Original Data Type':         odt,
        'Recommended Data Type':      rdt,
        'Original Units':             orig_u,
        'Recommended Units':          rec_u,
        'Regex Pattern':              regex,
        'Explanation of Recommendation': old_exp,
        'Better Explanation':         better_exp,
    })

out_df = pd.DataFrame(out_rows)

# ---------------------------------------------------------------------------
# Summary of changes made
# ---------------------------------------------------------------------------
changed = [(i, g(df.iloc[i], 'Change Type'), CT_OVERRIDES[i])
           for i in CT_OVERRIDES]
print(f"Change Type corrections applied: {len(changed)}")
for idx, old, new in changed:
    orig = g(df.iloc[idx], 'Original Attribute Name') or '(new attr)'
    print(f"  Row {idx:3d}  [{old}] -> [{new}]   ({orig})")

rows_with_exp = sum(1 for r in out_rows if r['Better Explanation'])
print(f"\nBetter Explanation added for: {rows_with_exp} / {len(out_rows)} rows")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
ts = datetime.date.today().strftime("%Y%m%d")
out_path = save_csv(out_df, rf'C:\Users\Ravi\Documents\Limit_Switches_schema_corrected_{ts}.csv')
print(f"\nSaved: {out_path}")
