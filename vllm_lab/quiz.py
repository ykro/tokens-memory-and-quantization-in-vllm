"""Ten short questions with one exact answer each, to compare a bf16 and a quantized model.

Short, closed answers keep the grading deterministic: normalize the text (lowercase, no
accents) and look for one of the accepted answers as a whole word. Ten questions only
catch large drops in quality; they do not replace a real evaluation of your use case.
"""

from __future__ import annotations

import re
import unicodedata

QUIZ: list[tuple[str, tuple[str, ...]]] = [
    ("¿Cuánto es 17 + 25? Responde solo con el número.", ("42",)),
    ("¿Cuál es la capital de Guatemala? Responde solo con el nombre.", ("guatemala",)),
    ("¿Cuántos días tiene una semana? Responde solo con el número.", ("7", "siete")),
    ("¿Cuánto es 9 por 8? Responde solo con el número.", ("72",)),
    ("¿Qué planeta es conocido como el planeta rojo? Responde con una palabra.", ("marte",)),
    ("¿Cuántos lados tiene un hexágono? Responde solo con el número.", ("6", "seis")),
    ("¿Cuál es el símbolo químico del oro? Responde solo con el símbolo.", ("au",)),
    ("¿Cuánto es 100 dividido entre 4? Responde solo con el número.", ("25",)),
    (("¿Qué lenguaje de programación usa la palabra def para definir funciones? "
      "Responde con una palabra."), ("python",)),
    ("¿Cuántos minutos tiene una hora? Responde solo con el número.", ("60", "sesenta")),
]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def is_correct(answer: str, accepted: tuple[str, ...]) -> bool:
    words = set(re.findall(r"\w+", normalize(answer)))
    return any(a in words for a in accepted)
