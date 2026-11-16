"""Fallback text when no LLM is configured. Telugu and Hindi wording needs native-speaker review."""
from .synth import MONTHS


def reason(name, p):
    e = p["series"]
    t = (f"{name}'s eGFR moved from {e[0]:.0f} to {e[-1]:.0f} over {p['months'][-1]} months, "
         f"a decline of about {p['decline']:.1f} mL/min/1.73 m2 per year. ")
    t += ("KDIGO describes a sustained fall of more than 5 per year as rapid progression. " if p["decline"] > 5
          else "That is faster than a typical age-related drift. " if p["decline"] > 3 else "That is close to a stable trend. ")
    if p["albuminuria"] != "A1":
        t += f"Urine albumin is in the {p['albuminuria']} category, which adds risk. "
    if p["alarm_idx"] >= 0:
        t += f"The CUSUM check flagged a change point at month {p['months'][p['alarm_idx']]}. "
    return t.strip()


WHEN = {"en": ("this week", "this month"), "te": ("ఈ వారం", "ఈ నెలలో"), "hi": ("इस हफ्ते", "इस महीने")}
MSG = {
    "en": "Hello {n}, your recent kidney tests show some changes. Please see your doctor {w} for a follow-up check. We are here to help you book it.",
    "te": "నమస్కారం {n}, మీ ఇటీవలి కిడ్నీ పరీక్షల్లో కొన్ని మార్పులు కనిపించాయి. దయచేసి {w} మీ డాక్టర్‌ను కలిసి పరీక్ష చేయించుకోండి. అపాయింట్‌మెంట్ తీసుకోవడంలో మేము సహాయం చేస్తాము.",
    "hi": "नमस्ते {n}, आपकी हाल की किडनी जाँच में कुछ बदलाव दिखे हैं। कृपया {w} अपने डॉक्टर से मिलकर जाँच करवाएँ। अपॉइंटमेंट लेने में हम आपकी मदद करेंगे।",
}


def draft(lang, first, tier):
    return MSG[lang].format(n=first, w=WHEN[lang][0 if tier == "High" else 1])
