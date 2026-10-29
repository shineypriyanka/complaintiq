import re

# Order matters: specific patterns first, the generic number pattern last.
PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[PAN]"),               # Indian PAN card
    (re.compile(r"(?:\+91[\s-]?)?\b[6-9]\d{9}\b"), "[PHONE]"),       # Indian mobile
    (re.compile(r"\b(?:\d[ -]?){6,19}\b"), "[NUMBER]"),              # card/account/Aadhaar
]

def redact(text: str) -> tuple[str, dict[str, int]]:
    counts = {}
    for pattern, tag in PATTERNS:
        text, n = pattern.subn(tag, text)
        if n:
            counts[tag] = n
    return text, counts