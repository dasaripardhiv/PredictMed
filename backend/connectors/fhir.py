"""STARTER for a FHIR R4 server (not tested against a real server).
Pulls creatinine Observations for one patient and converts them to eGFR with CKD-EPI 2021.
Confirm LOINC codes and units with your lab/EHR team before relying on this."""
import os
import httpx
from app.trajectory import ckd_epi_2021

CREATININE_LOINC = "2160-0"  # Creatinine [Mass/volume] in Serum or Plasma


def egfr_series(patient_id: str, age: float, female: bool, n: int = 8):
    base = os.environ["FHIR_BASE_URL"]  # e.g. https://fhir.example.org/r4
    headers = {"Authorization": f"Bearer {os.environ['FHIR_TOKEN']}"} if os.getenv("FHIR_TOKEN") else {}
    r = httpx.get(f"{base}/Observation", headers=headers, timeout=30,
                  params={"patient": patient_id, "code": f"http://loinc.org|{CREATININE_LOINC}",
                          "_sort": "date", "_count": 50})
    r.raise_for_status()
    vals = []
    for e in r.json().get("entry", []):
        q = e["resource"].get("valueQuantity")
        if q and q.get("unit") in ("mg/dL", "mg/dl"):  # TODO: convert umol/L (divide by 88.4)
            vals.append(ckd_epi_2021(q["value"], age, female))
    return vals[-n:]
