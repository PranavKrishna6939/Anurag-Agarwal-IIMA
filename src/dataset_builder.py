"""Reproducible 60-case Indian Infrastructure Arbitration benchmark corpus builder.

Models realistic Section 34 / Section 37 Arbitration & Conciliation Act (1996)
infrastructure disputes across 6 sectors (Highways, Railways/Metro, Power,
Ports, Aviation, Water/Urban EPC) with gold entity spans and domain keywords
so the complete RAG-NER, clustering, and award-estimation pipeline runs
deterministically offline without paid API keys.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np


SECTOR_PROFILES: Dict[str, dict] = {
    "Highways & Roads": {
        "respondents": ["National Highways Authority of India (NHAI)", "Ministry of Road Transport and Highways (MoRTH)", "Maharashtra State Road Development Corp."],
        "claimants": ["IRB Infrastructure Developers Ltd.", "Dilip Buildcon Ltd.", "Ashoka Buildcon Ltd.", "Sadbhav Engineering Ltd.", "PNC Infratech Ltd."],
        "keywords": [
            "toll plaza", "four-laning", "EPC highway", "right of way", "land acquisition delay",
            "prolongation cost", "NHAI concession", "HAM annuity", "utility shifting", "bitumen escalation",
            "independent engineer", "provisional completion certificate", "change of scope", "encumbrance-free land",
            "carriageway", "flyover substructure", "bot toll revenue", "force majeure", "idle machinery", "overhead recovery",
            "section 34 challenge", "extension of time", "price adjustment formula", "soil strata variation",
        ],
        "claim_range": (35.0, 260.0),
        "base_recovery": 0.48,
    },
    "Railways & Metro": {
        "respondents": ["Delhi Metro Rail Corporation (DMRC)", "Rail Vikas Nigam Ltd. (RVNL)", "Dedicated Freight Corridor Corp. (DFCCIL)", "Mumbai Metro Rail Corp. (MMRC)"],
        "claimants": ["Larsen & Toubro Heavy Civil", "Afcons Infrastructure Ltd.", "NCC Ltd.", "Tata Projects Ltd.", "HCC-Samsung JV"],
        "keywords": [
            "tunnel boring machine", "elevated viaduct", "metro depot", "signaling and telecom", "ballastless track",
            "underground station", "geotechnical stratum", "track doubling", "electrification OHE", "rolling stock interface",
            "DMRC general conditions", "safety clearance delay", "design approval lag", "variation order", "price variation clause",
            "bank guarantee encashment", "liquidated damages waiver", "sub-contractor claim", "site handover", "dewatering cost",
            "structural girder", "commissioner of railway safety", "arbitral tribunal award", "pendent lite interest",
        ],
        "claim_range": (45.0, 320.0),
        "base_recovery": 0.44,
    },
    "Power & Energy": {
        "respondents": ["NTPC Ltd.", "Power Grid Corporation of India (PGCIL)", "Damodar Valley Corporation (DVC)", "Solar Energy Corp. of India (SECI)"],
        "claimants": ["BHEL Industrial Systems", "Tata Power Solar Systems", "Kalpataru Projects International", "KEC International Ltd.", "Sterling and Wilson Renewable"],
        "keywords": [
            "thermal power plant", "solar PPA tariff", "transmission tower", "substation commissioning", "coal linkage disruption",
            "change in law", "safeguard duty", "boiler turbine generator", "grid synchronization", "flue gas desulfurization",
            "CERC regulatory order", "force majeure outage", "deemed generation", "transmission corridor", "transformer failure",
            "EPC turnkey package", "performance bank guarantee", "delayed payment surcharge", "capacity utilization factor", "auxiliary consumption",
            "hydroelectric headrace", "wind farm curtailment", "module price surge", "commissioning milestone",
        ],
        "claim_range": (40.0, 300.0),
        "base_recovery": 0.52,
    },
    "Ports & Maritime": {
        "respondents": ["Jawaharlal Nehru Port Authority (JNPA)", "Deendayal Port Authority", "Chennai Port Authority", "Paradip Port Authority"],
        "claimants": ["DP World Nhava Sheva", "Adani Ports Terminal JV", "ITD Cementation India Ltd.", "Van Oord India Dredging", "Simplex Infrastructures Ltd."],
        "keywords": [
            "capital dredging", "container terminal berth", "quay wall construction", "breakwater armor", "royalty revenue share",
            "minimum guaranteed throughput", "draft depth shortfall", "berthing delay", "marine piling", "cargo handling crane",
            "port tariff authority", "concession agreement termination", "hydrographic survey", "reclamation backfill", "tidal window restriction",
            "demurrage claim", "vessel turnaround", "channel siltation", "coastal regulation zone", "jetty fender system",
            "harbor basin", "offshore pipeline", "marine insurance claim", "port access road",
        ],
        "claim_range": (30.0, 240.0),
        "base_recovery": 0.41,
    },
    "Aviation & Airports": {
        "respondents": ["Airports Authority of India (AAI)", "Mumbai International Airport Ltd.", "Bangalore International Airport Ltd."],
        "claimants": ["GMR Airports Infrastructure", "Shapoorji Pallonji Engineering", "Nagarjuna Construction Co.", "Siemens Mobility Airport Systems"],
        "keywords": [
            "terminal building expansion", "runway resurfacing", "airfield ground lighting", "aero-bridge installation", "baggage handling system",
            "ATC tower modernization", "passenger boarding gate", "apron taxiway", "AERA tariff determination", "revenue share concession",
            "pandemic force majeure", "aviation security clearance", "radar calibration", "hangar steel truss", "runway strip grading",
            "commercial space lease", "escalation index", "defect liability period", "design-build-operate", "milestone liquidated damages",
            "airside access pass", "night curfew window", "HVAC chiller package", "fire fighting compliance",
        ],
        "claim_range": (25.0, 220.0),
        "base_recovery": 0.46,
    },
    "Water & Urban EPC": {
        "respondents": ["Delhi Jal Board (DJB)", "Brihanmumbai Municipal Corporation (BMC)", "Chennai Metro Water Supply", "UP Jal Nigam"],
        "claimants": ["VA Tech Wabag Ltd.", "Suez India Water Technologies", "Megha Engineering & Infrastructures", "SPML Infra Ltd."],
        "keywords": [
            "sewage treatment plant", "water desalination unit", "pipeline laying", "stormwater drainage", "pumping station",
            "municipal right of way", "effluent discharge norm", "overhead water tank", "micro-tunneling sewer", "hydro-testing pressure",
            "utility relocation", "environmental clearance", "raw water intake", "membrane filtration", "smart water metering",
            "billing certification delay", "retention money release", "escalation in steel", "monsoon suspension", "operation and maintenance",
            "urban flood mitigation", "canal lining", "borewell dewatering", "civil structural audit",
        ],
        "claim_range": (20.0, 180.0),
        "base_recovery": 0.49,
    },
}

FORUMS = [
    "High Court of Delhi",
    "High Court of Bombay",
    "Supreme Court of India",
    "High Court of Madras",
    "High Court of Karnataka",
    "Commercial Court, Ahmedabad",
]

STATUTES = [
    "Section 34, Arbitration and Conciliation Act, 1996",
    "Section 37, Arbitration and Conciliation Act, 1996",
    "Section 11(6), Arbitration and Conciliation Act, 1996",
    "Section 31(7), Arbitration and Conciliation Act, 1996",
]

CLAUSE_TYPES = [
    "Prolongation & Idle Resources",
    "Change of Scope & Variation",
    "Price Escalation & Change in Law",
    "Wrongful Termination & Liquidated Damages",
]


def build_arbitration_corpus(
    n_cases: int = 60,
    seed: int = 42,
    out_path: str = "data/indian_infra_arbitration_cases.json",
    force_rebuild: bool = False,
) -> List[dict]:
    """Generate or load the reproducible 60-case Indian Infrastructure Arbitration dataset."""
    if os.path.exists(out_path) and not force_rebuild:
        with open(out_path, "r", encoding="utf-8") as f:
            return json.load(f)

    rng = np.random.default_rng(seed)
    sectors = list(SECTOR_PROFILES.keys())
    cases: List[dict] = []

    for idx in range(n_cases):
        sector = sectors[idx % len(sectors)]
        prof = SECTOR_PROFILES[sector]

        claimant = str(rng.choice(prof["claimants"]))
        respondent = str(rng.choice(prof["respondents"]))
        forum = str(rng.choice(FORUMS))
        statute = str(rng.choice(STATUTES))
        clause = str(rng.choice(CLAUSE_TYPES))
        year = int(rng.integers(2017, 2025))

        lo, hi = prof["claim_range"]
        claim_cr = round(float(rng.uniform(lo, hi)), 2)
        delay_months = int(rng.integers(4, 44))
        interest_rate = round(float(rng.choice([8.0, 9.0, 9.5, 10.0, 10.5, 11.0, 12.0])), 1)
        tribunal_unanimous = int(rng.random() < 0.78)
        respondent_delay_share = round(float(rng.uniform(0.25, 0.92)), 2)
        has_escalation_clause = int(rng.random() < 0.62)
        counterclaim_cr = round(float(claim_cr * rng.uniform(0.0, 0.28)), 2)
        evidence_strength = round(float(rng.uniform(0.35, 0.95)), 2)

        # Realistic underlying economic award-generation structure (INR Crores)
        clause_bonus = {
            "Price Escalation & Change in Law": 0.08,
            "Change of Scope & Variation": 0.05,
            "Prolongation & Idle Resources": -0.02,
            "Wrongful Termination & Liquidated Damages": -0.05,
        }[clause]

        recovery_ratio = (
            prof["base_recovery"]
            + 0.22 * (respondent_delay_share - 0.55)
            + 0.14 * (evidence_strength - 0.65)
            + 0.04 * has_escalation_clause
            + 0.03 * tribunal_unanimous
            + clause_bonus
            - 0.16 * (counterclaim_cr / max(claim_cr, 1.0))
            + 0.002 * min(delay_months, 36)
        )
        recovery_ratio = float(np.clip(recovery_ratio, 0.12, 0.88))
        noise_cr = float(rng.normal(0.0, 7.5))
        award_cr = round(float(np.clip(claim_cr * recovery_ratio + noise_cr, 2.5, claim_cr * 0.95)), 2)

        # Sample 21-24 domain keywords per case -> ~1,300+ total keyword occurrences across 60 cases
        n_kw = int(rng.integers(20, 24))
        chosen_kws = list(rng.choice(prof["keywords"], size=min(n_kw, len(prof["keywords"])), replace=False))

        # Construct realistic multi-section arbitral judgment text for RAG chunking & NER
        text = (
            f"IN THE {forum.upper()} | O.M.P. (COMM) No. {110 + idx}/{year}\n"
            f"BETWEEN: {claimant} (Claimant) AND {respondent} (Respondent)\n"
            f"STATUTORY PROVISION: {statute}\n"
            f"INFRASTRUCTURE SECTOR: {sector}\n\n"
            f"1. BACKGROUND & DISPUTE SUMMARY: The Claimant executed a major {sector.lower()} contract "
            f"involving {', '.join(chosen_kws[:6])}. Disputes arose concerning {clause.lower()} and "
            f"project execution delays spanning {delay_months} months, of which {int(respondent_delay_share*100)}% "
            f"was attributable to employer-side clearances, {', '.join(chosen_kws[6:12])}.\n\n"
            f"2. CLAIMS & COUNTERCLAIMS: The Claimant filed total claims of INR {claim_cr:.2f} Crore "
            f"covering {', '.join(chosen_kws[12:18])}. The Respondent raised a counterclaim of INR {counterclaim_cr:.2f} Crore "
            f"citing liquidated damages and performance issues around {', '.join(chosen_kws[18:])}. "
            f"Contractual price escalation clause applicable: {'Yes' if has_escalation_clause else 'No'}.\n\n"
            f"3. TRIBUNAL FINDINGS & AWARD: After examining contemporaneous site logs and independent engineer "
            f"certificates (evidentiary weight {evidence_strength:.2f}), the "
            f"{'unanimous' if tribunal_unanimous else 'majority'} three-member Arbitral Tribunal allowed "
            f"a net arbitral award of INR {award_cr:.2f} Crore with pendente lite and future interest at "
            f"{interest_rate:.1f}% per annum under {statute}."
        )

        # Gold entity annotations (~35 entities per case -> 2,100+ entities across 60 cases)
        gold_entities = [
            {"type": "CLAIMANT", "text": claimant},
            {"type": "RESPONDENT", "text": respondent},
            {"type": "SECTOR", "text": sector},
            {"type": "FORUM", "text": forum},
            {"type": "STATUTE", "text": statute},
            {"type": "CLAUSE_TYPE", "text": clause},
            {"type": "CLAIM_AMOUNT_CR", "text": f"INR {claim_cr:.2f} Crore"},
            {"type": "COUNTERCLAIM_CR", "text": f"INR {counterclaim_cr:.2f} Crore"},
            {"type": "AWARD_AMOUNT_CR", "text": f"INR {award_cr:.2f} Crore"},
            {"type": "INTEREST_RATE", "text": f"{interest_rate:.1f}%"},
            {"type": "DELAY_MONTHS", "text": f"{delay_months} months"},
        ] + [{"type": "DOMAIN_TERM", "text": kw} for kw in chosen_kws]

        cases.append({
            "case_id": f"ARB-IND-{year}-{idx + 1:03d}",
            "year": year,
            "sector": sector,
            "claimant": claimant,
            "respondent": respondent,
            "forum": forum,
            "statute": statute,
            "clause_type": clause,
            "claim_amount_cr": claim_cr,
            "counterclaim_cr": counterclaim_cr,
            "award_amount_cr": award_cr,
            "interest_rate": interest_rate,
            "delay_months": delay_months,
            "respondent_delay_share": respondent_delay_share,
            "has_escalation_clause": has_escalation_clause,
            "tribunal_unanimous": tribunal_unanimous,
            "evidence_strength": evidence_strength,
            "keywords": chosen_kws,
            "gold_entities": gold_entities,
            "text": text,
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)
    return cases
