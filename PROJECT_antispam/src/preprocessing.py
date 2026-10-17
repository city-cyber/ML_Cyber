"""Shared RFC 822/MIME preprocessing for training and inference."""

import hashlib
import re
import unicodedata
import warnings
from email import policy
from email.parser import BytesParser

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

PREPROCESSING_VERSION = "1"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def fingerprint(text: str) -> str:
    return hashlib.sha256(normalize(text).casefold().encode("utf-8")).hexdigest()


def decode_part(part) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        value = part.get_payload()
        return value if isinstance(value, str) else ""
    charset = part.get_content_charset()
    for encoding in dict.fromkeys([charset, "utf-8", "cp1252", "latin-1"]):
        if not encoding:
            continue
        try:
            return payload.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return payload.decode("utf-8", errors="replace")


def body_parts(message):
    """Не читаем вложенные письма и содержимое приложений."""
    if message.get_content_disposition() == "attachment" or message.get_filename():
        return [], [], 1
    if message.get_content_maintype() == "message":
        return [], [], 1
    if message.is_multipart():
        plain, html, attachments = [], [], 0
        for part in message.iter_parts():
            p, h, count = body_parts(part)
            plain.extend(p)
            html.extend(h)
            attachments += count
        return plain, html, attachments
    content_type = message.get_content_type()
    if content_type == "text/plain":
        return [decode_part(message)], [], 0
    if content_type == "text/html":
        return [], [decode_part(message)], 0
    return [], [], 0


def parse_email(raw: bytes) -> dict:
    """Одинаковое преобразование обучающих писем и новых .eml.

    В модель поступают только тема и тело. Служебные X-Spam-*, Received,
    From, имена файлов и название источника в текст не включаются.
    """
    message = BytesParser(policy=policy.default).parsebytes(raw)
    subject = normalize(str(message.get("Subject", "")))
    plain, html, attachments = body_parts(message)
    html_source = "\n".join(html)
    if any(part.strip() for part in plain):
        body = "\n".join(plain)
    elif html_source:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
            soup = BeautifulSoup(html_source, "html.parser")
        for element in soup(["script", "style", "noscript", "head"]):
            element.decompose()
        body = soup.get_text(" ", strip=True)
    else:
        body = ""
    body = normalize(body)
    text = normalize(f"{subject} {body}")
    letters = sum(char.isalpha() for char in text)
    return {
        "subject": subject, "body": body, "text": text,
        "message_id": str(message.get("Message-ID", "")).strip().casefold(),
        "raw_hash": hashlib.sha256(raw).hexdigest(),
        "text_hash": fingerprint(text),
        "body_hash": fingerprint(body if len(body) >= 30 else text),
        "n_chars": len(text), "n_words": len(text.split()),
        "subject_length": len(subject),
        "n_urls": len(re.findall(r"https?://|www\.", body + " " + html_source, re.I)),
        "n_exclamations": text.count("!"),
        "uppercase_ratio": sum(char.isupper() for char in text) / max(letters, 1),
        "digit_ratio": sum(char.isdigit() for char in text) / max(len(text), 1),
        "html": bool(html), "attachments": attachments,
        "parse_defects": len(message.defects),
        "replacement_chars": text.count("\ufffd"),
    }


