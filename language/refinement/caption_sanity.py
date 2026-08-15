"""Deterministic caption grammar and filler cleanup — no second model rewrite."""

from __future__ import annotations

import re

# Artificial spatial / hedging filler that must not appear in final captions.
_FILLER_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bsits close enough to matter to the action\b",
        r"\bclose enough to matter(?: to the action)?\b",
        r"\bstays? close to the main action\b",
        r"\bappears to be\b",
        r"\bseems to be\b",
        r"\blikely\b",
        r"\bprobably\b",
        r"\bpossibly\b",
        r"\bmatters in the moment\b",
        r"\bplays a clear role in the moment\b",
        r"\bbelongs among the defining details of the view\b",
        r"\bthe sort of\b[^.]*",
        r"\bshare the surrounding space\b",
        r"\bspecific enough to ground the story\b",
        r"\bcompact and shaped for daily work\b",
        r"\breads clearly as\b",
    )
)

# Entire sentences that are generic padding — drop, never rewrite into other fillers.
_BANNED_SENTENCE_RES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^The setting remains clearly indoors\.?$",
        r"^The setting is clearly indoors\.?$",
        r"^The setting is indoors\.?$",
        r"^It is daytime\.?$",
        r"^The main work underway is .+\.?$",
        r"^The scene centers on .+\.?$",
        r"^The activity centers on .+\.?$",
        r"^They are using .+\.?$",
        r"^The place is .+, specific enough to ground the story\.?$",
        r"^The room reads clearly as .+\.?$",
        r"^Attention stays fixed on .+\.?$",
        r"^Behind them,.+softens into the wider landscape\.?$",
        r"^Behind them,.+fall into the wider landscape\.?$",
        r"^A quiet exchange seems to pass .+\.?$",
        r"^The nearer person and horse stand closer to the camera than the pair farther back\.?$",
        r"^The nearer .+ stand closer to the camera than .+\.?$",
    )
)

_DOUBLE_ARTICLE = re.compile(
    r"\b(?:an|a)\s+(?:an|a)\s+",
    re.IGNORECASE,
)
_DOUBLE_THE = re.compile(r"\bthe\s+the\s+", re.IGNORECASE)
_DOUBLE_IS = re.compile(r"\bis\s+is\b", re.IGNORECASE)
_DOUBLE_WORD = re.compile(r"\b([a-z]{3,})\s+\1\b", re.IGNORECASE)
_AWKWARD_SOMEONE = re.compile(
    r"\bsomeone nearby farther back\b",
    re.IGNORECASE,
)
_FOREGROUND_REPEATS = re.compile(
    r"(?:\bin the foreground,?\s*){2,}",
    re.IGNORECASE,
)


def order_action_first_sentences(text: str) -> str:
    """People/actions lead; accessories and count stubs follow — no invented facts."""
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", (text or "").strip())
        if s.strip()
    ]
    if len(sentences) <= 1:
        return (text or "").strip()

    def _tier(sentence: str) -> int:
        lower = sentence.lower()
        prop_led = bool(
            re.match(
                r"^(?:a|an|the)\s+(?:\w+\s+){0,2}(?:bowl|cup|bottle|vase|book|phone|cell phone|sink|handbag|backpack|sports ball)\b",
                lower,
            )
        )
        if any(
            tok in lower
            for tok in (
                "riding",
                "leading",
                "playing",
                "skiing",
                "skateboarding",
                "holding a rope",
                "preparing",
                "working",
            )
        ):
            return 0
        if prop_led and not any(
            tok in lower
            for tok in ("riding", "leading", "playing", "wearing", "preparing")
        ):
            return 5
        if any(
            tok in lower
            for tok in ("person", "people", "man", "woman", "child", "girl", "boy")
        ):
            return 1
        if any(tok in lower for tok in ("fire", "smoke")):
            return 2
        if any(
            tok in lower
            for tok in ("holding", "beside", "next to", "behind", "in front", "around")
        ) or re.search(r"\bnear\b", lower):
            return 3
        if "readable text" in lower or 'reads "' in lower or "text reads" in lower:
            return 4
        if re.search(r"\b(?:two|three|four|\d+)\s+people are visible\b", lower):
            return 6
        if any(
            tok in lower
            for tok in ("handbag", "backpack", "suitcase", "sports ball", "ball sits", "ball rests")
        ) and not any(
            tok in lower for tok in ("person", "people", "riding", "playing")
        ):
            return 5
        return 4

    ranked = sorted(enumerate(sentences), key=lambda item: (_tier(item[1]), item[0]))
    ordered = [
        s if s.endswith((".", "!", "?")) else f"{s}."
        for _, s in ranked
    ]
    return " ".join(ordered)


def has_awkward_filler(text: str) -> bool:
    """True when caption still contains known artificial phrasing."""
    body = text or ""
    if _DOUBLE_ARTICLE.search(body) or _AWKWARD_SOMEONE.search(body):
        return True
    if any(pattern.search(body) for pattern in _FILLER_PATTERNS):
        return True
    for sentence in re.split(r"(?<=[.!?])\s+", body):
        s = sentence.strip()
        if any(pat.match(s) for pat in _BANNED_SENTENCE_RES):
            return True
    banned_fragments = (
        "setting remains clearly indoors",
        "main work underway",
        "share the surrounding space",
        "also visible in the scene",
        "they are using a keyboard",
        "a person talking to a person",
        "second person stands farther back in the frame",
    )
    lower = body.lower()
    return any(frag in lower for frag in banned_fragments)


def caption_sanity_score(text: str) -> float:
    """Higher is better. Used to prefer natural vs Ollama without LLM rewrite."""
    body = (text or "").strip()
    if not body:
        return 0.0
    score = min(1.0, len(body.split()) / 28.0)
    if has_awkward_filler(body):
        score -= 0.55
    if _DOUBLE_ARTICLE.search(body):
        score -= 0.35
    if re.search(r"\ba\s+skis\b|\ban\s+a\b", body, re.IGNORECASE):
        score -= 0.40
    # Broken fragments after accessory stripping / planner glitches.
    if re.search(r"\b(?:is|are)\s+also\s+visible\.?\s*$", body, re.IGNORECASE):
        score -= 0.55
    if re.search(r"(?:^|[.!?]\s+)(?:Is|Are)\s+also\s+visible\b", body):
        score -= 0.55
    if re.search(r"(?:^|[.!?]\s+)Close beside\b", body, re.IGNORECASE):
        score -= 0.35
    if re.search(r"\bgrazes?\b", body, re.IGNORECASE):
        score -= 0.25
    if "defines this" in body.lower() or "specific enough to ground" in body.lower():
        score -= 0.20
    if re.search(r"\b(?:is|are)\s+also\s+visible\b", body, re.IGNORECASE):
        score -= 0.30
    if re.search(r"\bUp close, the action is\b", body, re.IGNORECASE):
        score -= 0.35
    if re.search(r"\bmoves through the moment\b", body, re.IGNORECASE):
        score -= 0.45
    if re.search(r"\bSkis is\b", body):
        score -= 0.50
    # Reward clear subject+action with concrete nouns (prefer over thin template prose).
    if re.search(
        r"\b(?:cats?|dogs?|people|person|man|woman|car|truck)\b.{0,40}\b"
        r"(?:sleep|sleeps|sit|sits|stand|stands|walk|walks|cross|crossing|wear|wearing)\b",
        body,
        re.IGNORECASE,
    ):
        score += 0.35
    # Prefer subject-led captions over prop-led "Nearby is …" openers.
    if re.match(r"^nearby\b", body, re.IGNORECASE):
        score -= 0.20
    if re.match(
        r"^(a|an|the|\d+)\s+(person|man|woman|people|player|cat|dog|bowl|car)\b",
        body,
        re.IGNORECASE,
    ):
        score += 0.12
    elif re.search(r"\bperson\b|\bpeople\b|\bplayer\b|\bcats?\b", body, re.IGNORECASE):
        score += 0.08
    # Mild penalty for repeated label mentions.
    labels = re.findall(r"\b(?:baseball glove|glove|knife|bowl|sink|person)\b", body.lower())
    if labels:
        from collections import Counter

        top = Counter(labels).most_common(1)[0][1]
        if top >= 3:
            score -= 0.25
        elif top >= 2:
            score -= 0.10
    # Prefer captions that preserve quoted OCR / named signage.
    if re.search(r'["“][^"”]{2,}["”]', body):
        score += 0.18
    if re.search(r"\breads\b|\bsign\b|\btext\b", body, re.IGNORECASE):
        score += 0.08
    return max(0.0, score)


def fix_double_articles(text: str) -> str:
    """Collapse 'an a', 'a a', 'the the', 'is is', immediate word repeats."""
    if not text:
        return text
    updated = _DOUBLE_ARTICLE.sub("a ", text)
    updated = _DOUBLE_THE.sub("the ", updated)
    updated = _DOUBLE_IS.sub("is", updated)
    updated = _DOUBLE_WORD.sub(r"\1", updated)
    return updated


def strip_spatial_filler(text: str) -> str:
    """Remove artificial spatial filler phrases; keep surrounding grammar intact."""
    if not text:
        return text
    updated = text
    # Rewrite awkward compounds before generic filler removal.
    updated = _AWKWARD_SOMEONE.sub("a second person farther back", updated)
    updated = re.sub(
        r"\bsomeone nearby\b",
        "a second person",
        updated,
        flags=re.IGNORECASE,
    )
    # Drop redundant camera-depth comparisons once farther-back is already clear.
    updated = re.sub(
        r"(?:^|\s+)The nearer [^.]*closer to the camera than the pair farther back\.?",
        " ",
        updated,
        flags=re.IGNORECASE,
    )
    for pattern in _FILLER_PATTERNS:
        updated = pattern.sub("", updated)
    updated = _FOREGROUND_REPEATS.sub("In the foreground, ", updated)
    updated = re.sub(r"\s{2,}", " ", updated)
    updated = re.sub(r"\s+([,.;:!?])", r"\1", updated)
    updated = re.sub(r"\.\s*\.", ".", updated)
    updated = re.sub(r",\s*\.", ".", updated)
    return updated.strip()


def dedupe_object_mention_sentences(text: str) -> str:
    """Drop later sentences that only restate objects already named."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]
    if len(sentences) <= 1:
        return text.strip()
    kept: list[str] = []
    mentioned: set[str] = set()
    stop = {
        "with",
        "from",
        "that",
        "this",
        "into",
        "over",
        "under",
        "while",
        "where",
        "their",
        "there",
        "through",
        "near",
        "nearer",
        "nearby",
        "stands",
        "standing",
        "wears",
        "wearing",
        "sits",
        "sitting",
        "foreground",
        "background",
        "action",
        "matter",
        "enough",
        "close",
        "farther",
        "someone",
        "another",
        "person",
        "people",
    }
    seen_instance_phrases: set[str] = set()
    for sentence in sentences:
        lower = sentence.lower()
        # Secondary instances ("another horse", "second person") are new coverage,
        # not restatements of the first mention of that noun.
        instance_phrases = {
            m.group(0)
            for m in re.finditer(
                r"\b(?:another|second|third|additional)\s+[a-z]{3,}\b",
                lower,
            )
        }
        tokens = {
            t
            for t in re.findall(r"[a-z]{3,}", lower)
            if t not in stop
        }
        # Multi-word sports gear / objects
        for phrase in ("baseball glove", "sports ball", "tennis racket", "dining table"):
            if phrase in lower:
                tokens.add(phrase.replace(" ", "_"))
        # Treat first-seen "another X" phrases as unique content tokens.
        for phrase in instance_phrases:
            if phrase not in seen_instance_phrases:
                tokens.add(phrase.replace(" ", "_"))
        content = {t for t in tokens if t not in mentioned}
        if kept and tokens and not content and len(sentence.split()) <= 16:
            continue
        # Drop pure foreground restatements of a single already-mentioned object.
        if kept and "foreground" in lower and len(content) <= 1 and not instance_phrases:
            continue
        # Drop "In the foreground, a X." when X already mentioned.
        if kept and re.match(r"^in the foreground,?\s+", sentence, re.IGNORECASE):
            leftover = {
                t
                for t in tokens
                if t not in mentioned and t not in {"foreground", "background"}
            }
            if len(leftover) <= 1 and not instance_phrases:
                continue
        kept.append(sentence if sentence.endswith((".", "!", "?")) else sentence + ".")
        mentioned |= tokens
        seen_instance_phrases |= instance_phrases
    return " ".join(kept)


def drop_malformed_caption_sentences(text: str) -> str:
    """Remove broken VLM/template fragments; never invent replacements."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]
    if not sentences:
        return (text or "").strip()
    kept: list[str] = []
    for sentence in sentences:
        body = sentence if sentence.endswith((".", "!", "?")) else sentence + "."
        lower = body.lower()
        # "They are, a blue shirt..." — truncated copula + noun phrase.
        if re.match(
            r"^(?:they|he|she|it|we|you)\s+are,\s+(?:a|an|the)\b",
            lower,
        ):
            continue
        # "A person, gloves and riding a bike on the water." — inventory glue.
        if re.match(
            r"^(?:a|an|the)\s+person,\s+\w+(?:\s*,\s*\w+)*\s+and\s+"
            r"(?:riding|holding|leading|playing|walking|running|standing|carrying)\b",
            lower,
        ):
            continue
        # Orphan clause starting with lowercase connector after bad splits.
        if re.match(r"^(?:and|but|with|while),?\s+(?:a|an|the)\b", lower):
            continue
        kept.append(body)
    return " ".join(kept)


def fold_support_object_into_activity(text: str) -> str:
    """Fold a short object-presence sentence into the prior activity sentence.

    Example:
      "Two people are playing football. A white sports ball rests in the scene."
      → "Two people are playing football while a white ball lies nearby."
    Only folds when both clauses are already evidenced; never invents locations.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]
    if len(sentences) < 2:
        return (text or "").strip()

    _ACTIVITY = (
        r"riding|playing|leading|holding|holds|walking|running|carrying|"
        r"swinging|skiing|skating|skateboarding|working|typing|guiding"
    )
    support_re = re.compile(
        r"^(?:a|an|the)\s+((?:\w+\s+){0,3}\w+)\s+"
        r"(?:rests|sits|lies|is visible|appears|remains)\s+"
        r"(?:in|at|near|within)\s+the\s+scene\.?$",
        re.IGNORECASE,
    )
    out: list[str] = []
    i = 0
    while i < len(sentences):
        cur = sentences[i]
        cur_body = cur if cur.endswith((".", "!", "?")) else cur + "."
        if i + 1 < len(sentences):
            nxt = sentences[i + 1]
            nxt_body = nxt if nxt.endswith((".", "!", "?")) else nxt + "."
            m = support_re.match(nxt_body.strip())
            if m and re.search(rf"\b(?:{_ACTIVITY})\b", cur_body, re.IGNORECASE):
                noun = m.group(1).strip()
                # Soften detector jargon without inventing new objects.
                noun = re.sub(r"\bsports\s+ball\b", "ball", noun, flags=re.IGNORECASE)
                noun = re.sub(r"\s{2,}", " ", noun).strip()
                art = "an" if re.match(r"^[aeiou]", noun, re.IGNORECASE) else "a"
                # Preserve an explicit determiner already in the capture when present.
                if re.match(r"^(?:a|an|the)\s+", noun, re.IGNORECASE):
                    np = noun
                else:
                    np = f"{art} {noun}"
                lead = cur_body.rstrip(".!?")
                out.append(f"{lead} while {np} lies nearby.")
                i += 2
                continue
        out.append(cur_body)
        i += 1
    return " ".join(out)


def dedupe_semantic_facts(text: str) -> str:
    """Drop later sentences that restate the same action/object fact."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]
    if len(sentences) <= 1:
        return (text or "").strip()

    _SPATIAL = frozenset({"near", "beside", "nearby", "next", "alongside"})
    _ACTION_LEMMA = {
        "holding": "hold",
        "holds": "hold",
        "hold": "hold",
        "held": "hold",
        "leading": "lead",
        "leads": "lead",
        "lead": "lead",
        "led": "lead",
        "guiding": "lead",
        "guides": "lead",
        "guide": "lead",
        "riding": "ride",
        "rides": "ride",
        "ride": "ride",
        "playing": "play",
        "plays": "play",
        "play": "play",
        "carrying": "carry",
        "carries": "carry",
        "carry": "carry",
        "swinging": "swing",
        "swings": "swing",
        "working": "work",
        "typing": "type",
    }
    _RIDEABLE = frozenset(
        {"motorcycle", "motorbike", "bicycle", "bike", "scooter", "dirt"}
    )

    def _fact_keys(sentence: str) -> set[str]:
        lower = sentence.lower()
        keys: set[str] = set()
        # Action+object pairs that commonly get restated.
        _ACTION_VERBS = tuple(_ACTION_LEMMA.keys()) + (
            "using",
            "use",
            "typing",
            "working",
        )
        for obj in (
            "keyboard",
            "mouse",
            "laptop",
            "computer",
            "monitor",
            "skis",
            "snowboard",
            "bicycle",
            "motorcycle",
            "bike",
            "table",
            "chair",
            "bottle",
            "phone",
            "cup",
            "book",
            "bag",
            "rope",
            "horse",
        ):
            if obj in lower and any(a in lower for a in _ACTION_VERBS):
                keys.add(f"use:{obj}")
            if obj in lower and any(s in lower for s in _SPATIAL):
                keys.add(f"spatial:{obj}")
        if any(tok in lower for tok in ("indoor", "indoors", "office", "workspace")) and any(
            tok in lower for tok in ("setting", "remains", "clearly", "location is")
        ):
            keys.add("setting:indoor")
        # Synonymous person-intro restatements.
        if re.search(r"\b(?:a |an |the )?person\b", lower) and any(s in lower for s in _SPATIAL):
            keys.add("person:spatial")
        # Claim-level keys: same verified relation restated with different tense/articles.
        if re.search(
            r"\b(?:hold|holds|holding|held)\b(?:\s+\w+){0,4}\s+\brope\b"
            r"|\brope\b(?:\s+\w+){0,4}\s+\b(?:hold|holds|holding|held)\b",
            lower,
        ):
            keys.add("action:hold_rope")
        if re.search(
            r"\b(?:lead|leads|leading|led|guiding|guide|guides)\b(?:\s+\w+){0,4}\s+\bhorse\b"
            r"|\bhorse\b(?:\s+\w+){0,4}\s+\b(?:lead|leads|leading|led|guiding)\b",
            lower,
        ):
            keys.add("action:lead_horse")
        if re.search(r"\b(?:engaged in\s+)?leading\b", lower):
            keys.add("action:leading")
        if re.search(r"\brid(?:e|es|ing)\b", lower):
            keys.add("action:ride")
            if any(tok in lower for tok in _RIDEABLE) or "dirt bike" in lower:
                keys.add("action:ride_vehicle")
        return keys

    def _norm_sentence(sentence: str) -> str:
        return re.sub(r"[^a-z0-9 ]+", "", sentence.lower()).strip()

    def _main_action(sentence: str) -> str | None:
        lower = sentence.lower()
        for raw, lemma in _ACTION_LEMMA.items():
            if re.search(rf"\b{re.escape(raw)}\b", lower):
                return lemma
        return None

    kept: list[str] = []
    seen_facts: set[str] = set()
    seen_tokens: set[str] = set()
    seen_norms: set[str] = set()
    seen_actions: set[str] = set()
    for sentence in sentences:
        keys = _fact_keys(sentence)
        norm = _norm_sentence(sentence)
        if norm and norm in seen_norms:
            continue
        if keys and keys.issubset(seen_facts):
            continue
        action = _main_action(sentence)
        # Bare / shorter activity restatement: "A person is riding." after a full ride sentence.
        if (
            kept
            and action
            and action in seen_actions
            and len(sentence.split()) <= 6
            and keys.issubset(seen_facts | {f"action:{action}", "action:ride", "action:ride_vehicle"})
        ):
            continue
        if kept and action == "ride" and "action:ride" in seen_facts and len(sentence.split()) <= 8:
            # Drop "A person is riding a motorcycle." when dirt-bike/motorcycle already narrated.
            if "action:ride_vehicle" in seen_facts or any(
                tok in seen_tokens for tok in ("motorcycle", "motorbike", "bicycle", "bike", "dirt")
            ):
                continue
        # Near-duplicate spatial restatement (near vs beside same objects).
        if kept and "person:spatial" in keys and "person:spatial" in seen_facts:
            overlap = {t for t in re.findall(r"[a-z]{4,}", sentence.lower())} & seen_tokens
            if len(overlap) >= 2:
                continue
        # Drop activity restatement when lead already contains the activity nouns.
        if kept and re.match(
            r"^(?:The (?:main |observed )?(?:activity|work underway)|Up close, the (?:action|real work)) is\b",
            sentence,
            re.IGNORECASE,
        ):
            activity = re.sub(
                r"^(?:The (?:main |observed )?(?:activity|work underway)|Up close, the (?:action|real work)) is\s+",
                "",
                sentence,
                flags=re.IGNORECASE,
            )
            act_tokens = {t for t in re.findall(r"[a-z]{4,}", activity.lower())}
            if act_tokens and (
                act_tokens.issubset(seen_tokens) or any(t in seen_tokens for t in act_tokens)
            ):
                continue
        kept.append(sentence if sentence.endswith((".", "!", "?")) else sentence + ".")
        seen_facts |= keys
        seen_tokens |= {t for t in re.findall(r"[a-z]{4,}", sentence.lower())}
        if norm:
            seen_norms.add(norm)
        if action:
            seen_actions.add(action)
    return " ".join(kept)


def strip_subjective_language(text: str) -> str:
    """Remove narrative/psychological filler and multi-person activity over-attribution."""
    updated = (text or "").strip()
    if not updated:
        return updated

    # Formal / robotic phrasing → everyday language (before sentence drops).
    # IMPORTANT: rewrite "engaged in an activity" BEFORE bare "engaged in",
    # otherwise "are engaged in an activity" collapses to "are an activity".
    replacements = (
        (r"\ba pair of individuals\b", "two people"),
        (r"\bA pair of individuals\b", "Two people"),
        (r"\bA pair are\b", "Two people are"),
        (r"\ba pair are\b", "two people are"),
        (r"\bA pair is\b", "Two people are"),
        (r"\bone individual\b", "one person"),
        (r"\bOne individual\b", "One person"),
        (r"\bThis individual\b", "This person"),
        (r"\bindividuals are present\b", "people are"),
        (r"\bindividuals\b", "people"),
        (r"\boutdoor farm pasture setting\b", "grassy field"),
        (r"\bfarm pasture setting\b", "grassy field"),
        (r"\bfarm pasture\b", "grassy field"),
        (r"\bis situated in the background\b", "stands farther back"),
        (r"\bsituated in the background\b", "farther back"),
        (r"\bThe overall impression suggests\b", ""),
        (r"\boverall impression suggests\b", ""),
        (r"\bengaged in leading\b", "leading"),
        (r"\ban brown\b", "a brown"),
        (r"\ban grassy\b", "a grassy"),
        (r"\bactively\s+(leading|holding|riding|walking|guiding|carrying)\b", r"\1"),
        (r"\bactively\b", ""),
    )
    for pattern, repl in replacements:
        updated = re.sub(pattern, repl, updated, flags=re.IGNORECASE)

    # Fix engaged-in templates without destroying grammar.
    # Order matters: "engaged in an activity" before bare "engaged in",
    # otherwise "are engaged in an activity" collapses to "are an activity".
    updated = re.sub(
        r"\bare engaged in an activity(?:\s+(?:within|in))?\b",
        "are in",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\bis engaged in an activity(?:\s+(?:within|in))?\b",
        "is in",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\bTwo people are an activity(?:\s+(?:within|in))?\b",
        "Two people are in",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\bTwo people are an(?:\s+\w+){0,3}\s+activity(?:\s+(?:within|in|involving))?\b",
        "Two people are in",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\bA person is an activity(?:\s+(?:within|in))?\b",
        "A person is in",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\b(?:are|is)\s+an activity(?:\s+(?:within|in))?\b",
        "are in",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\b(?:are|is)\s+an(?:\s+\w+){0,3}\s+activity(?:\s+(?:within|in|involving))?\b",
        "are in",
        updated,
        flags=re.IGNORECASE,
    )
    # Remaining "engaged in <verb>" → keep the verb.
    updated = re.sub(
        r"\bare engaged in\s+([a-z]+ing)\b",
        r"are \1",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\bis engaged in\s+([a-z]+ing)\b",
        r"is \1",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(r"\bare engaged in\b", "are", updated, flags=re.IGNORECASE)
    updated = re.sub(r"\bis engaged in\b", "is", updated, flags=re.IGNORECASE)

    # Never attribute one person's verified activity to all people in the scene.
    updated = re.sub(
        r"\b(?:two|both)\s+people\s+(?:are\s+)?(?:both\s+)?(?:engaged in\s+)?leading\b",
        "one person is leading",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\b(?:two|both)\s+people\s+(?:are\s+)?(?:both\s+)?(?:engaged in\s+)?"
        r"(holding|riding|using|carrying|guiding)\b",
        r"one person is \1",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\btwo people(?: who are)?(?: both)? leading a horse\b",
        "one person leading a horse",
        updated,
        flags=re.IGNORECASE,
    )
    # Proximity must not become participation / observation — rewrite clauses first
    # so factual lead sentences are preserved rather than dropped wholesale.
    updated = re.sub(
        r"\bwhile (?:another|the other) person (?:is )?(?:observing|watching|helping with) "
        r"(?:the )?(?:activity|horse|scene)\b",
        "while another person stands farther back",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\b(?:who|that) (?:is |are )?(?:observing|watching) the activity\b",
        "farther back in the scene",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\b(?:nearby|near them|close by),?\s+(?:also\s+)?(?:helping|participating|talking|leading)\b",
        "farther back",
        updated,
        flags=re.IGNORECASE,
    )

    # Drop remaining subjective / atmospheric sentences.
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", updated) if p.strip()]
    kept: list[str] = []
    banned_sentence = re.compile(
        r"(?:"
        r"overall impression|"
        r"casual moment|"
        r"appears to be enjoying|"
        r"appears happy|"
        r"appears interested|"
        r"seems to be watching|"
        r"observing the activity|"
        r"observing the scene|"
        r"watching the activity|"
        r"watching the scene|"
        r"feels like|"
        r"the mood|"
        r"atmosphere (?:stays|is|feels)"
        r")",
        re.IGNORECASE,
    )
    for part in parts:
        sentence = part if part.endswith((".", "!", "?")) else part + "."
        if banned_sentence.search(sentence):
            continue
        kept.append(sentence)
    updated = " ".join(kept)

    updated = re.sub(r"\s{2,}", " ", updated).strip()
    updated = re.sub(r"\s+([,.;:!?])", r"\1", updated)
    # Clean empty leftovers after stripping "The overall impression suggests".
    updated = re.sub(r"^\s*[,;:\-–—]+\s*", "", updated)
    if updated and updated[0].islower():
        updated = updated[0].upper() + updated[1:]
    return updated


def normalize_animal_coat_color(color: str) -> str:
    """Map unreliable animal coat labels to safer everyday colors."""
    value = (color or "").strip().lower()
    remap = {
        "olive": "brown",
        "olive green": "brown",
        "khaki": "tan",
        "beige": "tan",
        "burgundy": "brown",
        "maroon": "brown",
        "navy": "dark brown",
        "navy blue": "dark brown",
        "teal": "gray",
        "purple": "dark brown",
        "pink": "light brown",
        "coral": "light brown",
        "mustard": "tan",
    }
    return remap.get(value, value)


def strip_unreliable_animal_colors(text: str) -> str:
    """Replace olive/burgundy/etc. horse coats with safer brown/tan wording."""
    updated = (text or "").strip()
    if not updated:
        return updated
    animal = r"(horse|dog|cat|cow|sheep|goat|pony|mane)"
    for bad, good in (
        ("olive green", "brown"),
        ("olive", "brown"),
        ("burgundy", "brown"),
        ("maroon", "brown"),
        ("navy blue", "dark brown"),
        ("navy", "dark brown"),
        ("khaki", "tan"),
        ("beige", "tan"),
    ):
        updated = re.sub(
            rf"\b{re.escape(bad)}\s+{animal}\b",
            rf"{good} \1",
            updated,
            flags=re.IGNORECASE,
        )
        updated = re.sub(
            rf"\b{re.escape(bad)}-colored\s+{animal}\b",
            rf"{good} \1",
            updated,
            flags=re.IGNORECASE,
        )
        updated = re.sub(
            rf"\b{re.escape(bad)}\s+colored\s+{animal}\b",
            rf"{good} \1",
            updated,
            flags=re.IGNORECASE,
        )
        updated = re.sub(
            rf"\b{animal}\s+(?:is|are|appears)\s+{re.escape(bad)}\b",
            rf"\1 is {good}",
            updated,
            flags=re.IGNORECASE,
        )
    return updated


def humanize_caption_style(text: str) -> str:
    """Rewrite robotic / abstract caption phrasing into plain visual English.

    Style-only: never invent objects or activities. Safe to run on any caption
    candidate before lock.
    """
    updated = (text or "").strip()
    if not updated:
        return updated

    # Full-sentence abstract reports → short visual English.
    full_sentence_rewrites = (
        (
            r"\bThe environment indicates a domestic food preparation setting\.?",
            "The scene is a kitchen.",
        ),
        (
            r"\bThe environment indicates a food preparation setting\.?",
            "The scene is a kitchen.",
        ),
        (
            r"\bThe animal occupies a natural environment\.?",
            "An animal stands outdoors.",
        ),
        (
            r"\bThe vehicle participates in urban transportation\.?",
            "A vehicle is on a road.",
        ),
        (
            r"\bThe motorcycle is situated in a transportation corridor setting,"
            r"\s*indicating it(?:['’]s| is) part of an outdoor traffic flow\.?",
            "A motorcycle is on a road.",
        ),
        (
            r"\bThe scene indicates part of an outdoor traffic flow\.?",
            "",
        ),
        (
            r"\bThe scene indicates part of[^.]+\.?",
            "",
        ),
    )
    for pattern, repl in full_sentence_rewrites:
        updated = re.sub(pattern, repl, updated, flags=re.IGNORECASE)

    # Abstract venue labels → simple place words (caption language only).
    place_rewrites = (
        (r"\btransportation corridor(?: setting)?\b", "road"),
        (r"\btraffic corridor\b", "road"),
        (r"\burban environment\b", "city street"),
        (r"\bnatural environment\b", "outdoor area"),
        (r"\brecreational (?:area|setting|environment)\b", "outdoor area"),
        (r"\bcommercial (?:area|setting|environment)\b", "street"),
        (r"\btraffic flow\b", "road"),
        (r"\boutdoor traffic flow\b", "road"),
        (r"\bfarm pasture(?: setting)?\b", "grassy field"),
        (r"\benvironment setting\b", "area"),
        (r"\bindoor room\b", "room"),
        (r"\boutdoor area\b", "outdoors"),
        (r"\bwithin a ([a-z][\w\s-]{0,40}?) setting\b", r"in a \1"),
        (r"\bin a ([a-z][\w\s-]{0,40}?) setting\b", r"in a \1"),
        (r"\bwithin a\b", "in a"),
        (r"\bwithin an\b", "in an"),
        (r"\bwithin the\b", "in the"),
    )
    for pattern, repl in place_rewrites:
        updated = re.sub(pattern, repl, updated, flags=re.IGNORECASE)

    # Clothing report style → natural "wearing …".
    updated = re.sub(
        r"\bdressed in\s+[\w\s-]{1,40}?\sattire\s*[-–—,:]\s*"
        r"(.+?)(?:\s+paired with\s+[^.,;]+?)?(?:\s*[-–—]\s*)?(?=\s*(?:is|are)\b)",
        r"wearing \1 ",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\bdressed in\s+([\w\s-]{1,40}?)\sattire\b",
        r"wearing \1 clothing",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\bwearing\s+([^.,;]+?)\s+paired with\s+([^.,;]+?)(?=\s*[-–—,]|\s+(?:is|are)\b|[.])",
        r"wearing \1 and \2",
        updated,
        flags=re.IGNORECASE,
    )
    # Remove dangling dash before the verb after a clothing clause.
    updated = re.sub(
        r"(wearing [^.,;]+?)\s*[-–—]\s*(is|are)\b",
        r"\1 \2",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(r",\s*(wearing\b)", r" \1", updated, flags=re.IGNORECASE)

    # Robotic verbs / report style → human description.
    style_rewrites = (
        (
            r"\bis situated (?:in|on) (?:a |an |the )?(road|street|highway|path|field|bridge)\b",
            r"is on a \1",
        ),
        (
            r"\bare situated (?:in|on) (?:a |an |the )?(road|street|highway|path|field|bridge)\b",
            r"are on a \1",
        ),
        (r"\bis situated in\b", "is in"),
        (r"\bare situated in\b", "are in"),
        (r"\bsituated in (?:a |an |the )?(road|street|highway|path|field)\b", r"on a \1"),
        (r"\bsituated in\b", "in"),
        (
            r"\bis positioned (?:within|in|on) (?:a |an |the )?(road|street|highway|path|field)\b",
            r"is on a \1",
        ),
        (r"\bis positioned (?:within|in|on)\b", "is in"),
        (r"\bare positioned (?:within|in|on)\b", "are in"),
        (r"\bis positioned near\b", "is near"),
        (r"\bare positioned near\b", "are near"),
        (r"\bis positioned beside\b", "is beside"),
        (r"\bare positioned beside\b", "are beside"),
        (r"\boccupies a\b", "stands in a"),
        (r"\boccupy a\b", "stand in a"),
        (r"\bparticipates in urban transportation\b", "is on a road"),
        (r"\bparticipate in urban transportation\b", "are on a road"),
        (r"\bparticipates in\b", "is in"),
        (r"\bparticipate in\b", "are in"),
        (r"\bis located within\b", "is in"),
        (r"\bare located within\b", "are in"),
        (r"\blocated in proximity to\b", "near"),
        (r"\bin proximity to\b", "near"),
        (r"\bpositioned adjacent to\b", "beside"),
        (r"\badjacent to\b", "beside"),
        (r"\bspatially related to\b", "near"),
        (r"\bsits within the scene\b", "is nearby"),
        (r"\bsit within the scene\b", "are nearby"),
        (r"\bsits within\b", "is in"),
        (r"\bsit within\b", "are in"),
        # Scene-graph / zone metadata must not leak into prose.
        (r"\bat the bottom center of (?:an |a |the )?outdoors\b", "outdoors"),
        (r"\bat the (?:bottom|top|middle) center of the scene\b", "in the scene"),
        (r"\bat the (?:bottom|top|middle)-(?:left|right|center) of the scene\b", "in the scene"),
        (r"\brests at the bottom center of the scene\b", "is nearby"),
        (r"\bis positioned at the (?:bottom|top|middle) center\b", "is"),
        (r"\bis positioned at the\b", "is at the"),
        (r"\bin an outdoors\b", "outdoors"),
        (r"\bin a outdoors\b", "outdoors"),
        (r"\bof an outdoors\b", "outdoors"),
        (r"\bacross an outdoors\b", "outdoors"),
        (r"\bthrough an outdoors\b", "outdoors"),
        (r"\ban outdoors\b", "outdoors"),
        (r"\ba outdoors\b", "outdoors"),
        (r"\bin outdoors\b", "outdoors"),
        (r"\bas evidenced by\b", "with"),
        (r"\bengaged in an activity\b", "active"),
        (r"\bengaged in\s+(playing|riding|leading|holding|working|walking|running|carrying)\b", r"\1"),
        (r"\bactively\s+(leading|holding|riding|walking|guiding|carrying)\b", r"\1"),
        (r"\bactively\b", ""),
        (r"\bcan be observed\b", "is visible"),
        (r"\bcan be seen to be\b", "is"),
        (r"\bThe scene depicts\b", ""),
        (r"\bThe scene represents\b", ""),
        (r"\bThe scene indicates\b", ""),
        (r"\bThe environment indicates\b", ""),
        (r"\benvironmental context indicates\b", ""),
        (r"\bThe overall scene shows\b", ""),
        (r"\bin a road setting\b", "on a road"),
        (r"\bin a road\b", "on a road"),
        (r"\bon a road setting\b", "on a road"),
        (r"\ba road setting\b", "a road"),
        (r"\bdomestic food preparation setting\b", "kitchen"),
        (r"\bfood preparation setting\b", "kitchen"),
        (r"\bfood preparation activity\b", "preparing food"),
        (r"\btransportation activity(?: detected)?\b", "riding"),
        (r"\binteraction with (?:a |an |the )?horse(?: detected)?\b", "with a horse"),
        (r"\bThe vehicle participates in urban transportation\b", "A vehicle is on a road"),
        (r"\bThe animal occupies a natural environment\b", "An animal stands outdoors"),
        (r"\ba solitary\b", "a"),
        (r"\bA solitary\b", "A"),
        # Gender-neutralization leftovers → natural plurals.
        (r"\b((?:[Aa]n?|[Tt]he)\s+)?person(?:\s*,)?\s+and\s+(?:(?:an?|the)\s+)?person\b", r"two people"),
        (r"\bare present in (?:an |a )?indoor\b", "are in an indoor"),
        (r"\bare present in\b", "are in"),
        (r"\bis present in\b", "is in"),
    )
    for pattern, repl in style_rewrites:
        updated = re.sub(pattern, repl, updated, flags=re.IGNORECASE)

    # Capitalize lead after person-and-person → two people rewrite.
    updated = re.sub(r"(^|[.!?]\s+)two people\b", lambda m: f"{m.group(1)}Two people", updated)
    updated = re.sub(
        r"(^|[.!?]\s+)(\d+)\s+people\b",
        lambda m: f"{m.group(1)}{ {'2':'Two','3':'Three','4':'Four','5':'Five','6':'Six'}.get(m.group(2), m.group(2)) } people",
        updated,
    )
    # Drop trailing census filler when plurality is already established earlier.
    trailing_census = re.search(
        r"\s+((?:Two|Three|Four|Five|Six|\d+)\s+people are visible(?: in the scene)?\.?)\s*$",
        updated,
        flags=re.IGNORECASE,
    )
    if trailing_census:
        body = updated[: trailing_census.start()].rstrip()
        if re.search(
            r"\b(?:two|three|four|several|both)\s+people\b|\bpeople\b|\banother person\b",
            body,
            re.I,
        ):
            updated = body if body.endswith((".", "!", "?")) else body + "."


    # Singular article + plural verb leftovers from support templates.
    def _fix_singular_are(match: re.Match[str]) -> str:
        subj = match.group(1)
        if " and " in subj.lower() or re.match(r"^\d+\s", subj.strip(), re.I):
            return match.group(0)
        if re.match(
            r"^(?:two|three|four|five|six|several|many|both)\b",
            subj.strip(),
            re.I,
        ):
            return match.group(0)
        return f"{subj} is {match.group(2)}"

    updated = re.sub(
        r"\b((?:A|An|The)(?!\s+(?:two|three|four|five|six)\b)[^.!?]{1,80}?) are (visible|nearby|arranged)\b",
        _fix_singular_are,
        updated,
    )
    # Drop orphan color-only visibility sentences.
    updated = re.sub(
        r"(?:^|[.!?]\s+)(?:A|An|The)\s+(?:light|dark|sky|navy|royal)?\s*"
        r"(?:blue|red|green|brown|beige|white|black|gray|grey|yellow|orange|"
        r"pink|purple|cream|tan|olive)\s+(?:is|are)\s+visible[^.!?]*[.!?]?\s*",
        " ",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(r"\s{2,}", " ", updated).strip()

    # Collapse broken "are/is an activity" leftovers (after other rewrites).
    updated = re.sub(
        r"\bTwo people are an(?:\s+\w+){0,3}\s+activity(?:\s+(?:within|in|involving))?\b",
        "Two people are in",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\bA person is an(?:\s+\w+){0,3}\s+activity(?:\s+(?:within|in|involving))?\b",
        "A person is in",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\b(?:are|is)\s+an(?:\s+\w+){0,3}\s+activity(?:\s+(?:within|in|involving))?\b",
        "are in",
        updated,
        flags=re.IGNORECASE,
    )
    # "Two people are an outdoor activity involving a horse" → natural lead.
    updated = re.sub(
        r"\b(?:Two|Both)\s+people\s+are\s+in\s+(?:a\s+)?(?:\w+\s+){0,2}involving\s+(a|an|the)\s+",
        r"Two people are with \1 ",
        updated,
        flags=re.IGNORECASE,
    )
    # Robotic attribute prose.
    updated = re.sub(
        r"\b(?:The|A|An)\s+(\w+(?:\s+\w+){0,3})['’]s\s+dominant\s+color\s+is\s+(\w+)\b",
        r"The \1 appears \2",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\b(?:the|a|an)\s+object['’]?s?\s+dominant\s+color\s+is\s+(\w+)\b",
        r"it appears \1",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\bdominant\s+color\s+is\s+(\w+)\b",
        r"color is \1",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\bhas a dominant color of\s+(\w+)\b",
        r"appears \1",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\bwith a dominant color of\s+(\w+)\b",
        r"that appears \1",
        updated,
        flags=re.IGNORECASE,
    )

    # Drop interpretive / purpose clauses (assumptions, not visible facts).
    updated = re.sub(
        r",?\s*indicating (?:that )?(?:it(?:['’]s| is) )?part of[^.]*",
        "",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r",?\s*indicating (?:that )?[^.,;]+",
        "",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r",?\s*suggesting (?:that )?[^.,;]+",
        "",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r",?\s*which (?:suggests|indicates|implies)[^.,;]+",
        "",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\b(?:this|that) (?:is )?likely (?:a |an )?[^.,;]+",
        "",
        updated,
        flags=re.IGNORECASE,
    )

    # Activity report openers → natural progressive verbs when leftover.
    updated = re.sub(
        r"\bThe observed activity is\s+([a-z][a-z\s-]{2,40})\b",
        lambda m: f"Someone is {m.group(1).strip()}",
        updated,
        flags=re.IGNORECASE,
    )

    # Cleanup artifacts from clause stripping.
    updated = re.sub(r"\s{2,}", " ", updated)
    updated = re.sub(r"\s+([,.;:!?])", r"\1", updated)
    updated = re.sub(r",[,\s]*\.", ".", updated)
    updated = re.sub(r"\.\s*\.", ".", updated)
    updated = re.sub(r"^\s*[,;:\-–—]\s*", "", updated)
    updated = updated.strip()
    if updated and updated[0].islower():
        updated = updated[0].upper() + updated[1:]
    # Ensure terminal punctuation if we still have a sentence.
    if updated and updated[-1] not in ".!?":
        updated = updated.rstrip(",;:") + "."
    return updated


def sanitize_caption(text: str) -> str:
    """Apply all safe deterministic caption cleanups."""
    updated = (text or "").strip()
    if not updated:
        return updated
    # Everyday language: drop stiff academic openers when they lead the caption.
    updated = re.sub(
        r"^(?:The image (?:depicts|shows|features|illustrates|presents)|"
        r"The scene (?:depicts|illustrates|presents)|"
        r"This (?:image|photo|picture) (?:shows|depicts|features))\s+",
        "",
        updated,
        flags=re.IGNORECASE,
    )
    if updated and updated[0].islower():
        updated = updated[0].upper() + updated[1:]
    # Humanize robotic / abstract phrasing before other cleanups.
    updated = humanize_caption_style(updated)
    updated = strip_subjective_language(updated)
    updated = strip_unreliable_animal_colors(updated)
    # Metadata / detector leakage.
    updated = re.sub(r"(?i)\bObserved activity\s*:\s*", "", updated)
    updated = re.sub(r"(?i)\bConfirmed\s*:\s*", "", updated)
    updated = re.sub(r"(?i)\bEntity\s*:\s*\S+\s*", "", updated)
    updated = re.sub(r"(?i)\bThe location is\s+(?:a |an |the )?outdoor\.?", "The scene is outdoors.", updated)
    updated = re.sub(r"(?i)\bThe location is outdoor\b", "The scene is outdoors", updated)
    updated = re.sub(
        r"(?i)\b(?:A |The )?person,\s+and\s+(bicycle|motorcycle|horse)\b",
        r"A person and \1",
        updated,
    )
    # Robotic dominant-color inventory phrasing.
    updated = re.sub(
        r"(?i)\b(?:The|A|An)\s+(\w+(?:\s+\w+){0,3})['’]s\s+dominant\s+color\s+is\s+(\w+)\b",
        r"The \1 appears \2",
        updated,
    )
    updated = re.sub(
        r"(?i)(?:^|[.]\s*)(?:[^.]*?\bdominant\s+color\s+is\b[^.]*?)(?=[.!?]|$)",
        "",
        updated,
    )
    # Fix broken "with one is …" constructions from template fusion.
    updated = re.sub(
        r",\s*with one is\b",
        ". One of them is",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\bwith one is\b",
        "one of them is",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(r"\ba other people\b", "other people", updated, flags=re.IGNORECASE)
    updated = re.sub(r"\bA other people\b", "Other people", updated)
    # Deduplicate adjacent fire/smoke restatements.
    updated = re.sub(
        r"(Smoke and fire are visible nearby\.)\s+A small fire burns near the ground\.",
        r"\1",
        updated,
        flags=re.IGNORECASE,
    )
    updated = strip_spatial_filler(updated)
    # Grammar: activity labels missing articles / redundant place phrases.
    updated = re.sub(r"\bcrossing street\b", "crossing a street", updated, flags=re.IGNORECASE)
    updated = re.sub(
        r"\bcrossing a street on (?:a |an |the )?(city street|street)\b",
        r"crossing a \1",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\bAround them lies a\b",
        "The surrounding area includes a",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\bAround them stretches a\b",
        "The surrounding area opens as a",
        updated,
        flags=re.IGNORECASE,
    )
    # Rewrite redundant place restatements; keep scenic details as full sentences.
    updated = re.sub(
        r"(?<=\.)\s*The surrounding area includes a (?:city )?street,\s*with\s+([^.]+)\.",
        lambda m: f" {m.group(1).strip().rstrip('.').capitalize()}.",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"(?<=\.)\s*The surrounding area includes a (?:city )?street[^.]*\.",
        "",
        updated,
        flags=re.IGNORECASE,
    )
    # Fragment repair: "Trees lining …" → complete sentence.
    updated = re.sub(
        r"\bTrees lining the edge of the (?:street|view)\b",
        "Trees line the edge of the street",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\btrees lining the edge of the (?:street|view)\b",
        "trees line the edge of the street",
        updated,
        flags=re.IGNORECASE,
    )
    # Soften inventory phrasing into natural presence wording.
    updated = re.sub(
        r"\b([^.]+?)\s+(?:is|are)\s+also\s+visible(?:\s+in\s+the\s+scene)?\b",
        r"\1 can also be seen nearby",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\bgrazes?\s+(?:quietly\s+)?(?:farther\s+back|nearby)\b",
        "rests farther back",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\bClose beside\s+(a|an|the)\b",
        r"Beside \1",
        updated,
        flags=re.IGNORECASE,
    )
    # Plural equipment grammar and poetic detector leftovers.
    updated = re.sub(r"\bSkis is\b", "Skis are", updated)
    updated = re.sub(r"\bskis is\b", "skis are", updated)
    updated = re.sub(r"\bPoles is\b", "Poles are", updated)
    updated = re.sub(
        r"\bmoves through the moment(?: outdoors)?\b",
        "is visible outdoors",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\b([A-Za-z]+ weather) is visible outdoors\b",
        r"\1 covers the outdoor scene",
        updated,
        flags=re.IGNORECASE,
    )
    # Drop redundant activity restatement when skiing already stated.
    if re.search(r"\bski(?:ing|s)\b", updated, re.IGNORECASE):
        updated = re.sub(
            r"(?<=\.)\s*The main activity is ski(?:ing)?\.",
            "",
            updated,
            flags=re.IGNORECASE,
        )
    # Street / place repetition: "crossing a street on a city street".
    updated = re.sub(
        r"\b(crossing a (?:city )?street)\s+on a (?:city )?street\b",
        r"\1",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\bAround them lies a (?:city )?street\b[^.]*\.",
        "",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"(?<=\.)\s*The location is (?:a |an |the )?(?:city )?street\.",
        "",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\bsit nearby in the scene\b",
        "are nearby",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\bis also visible in the scene\b",
        "is nearby",
        updated,
        flags=re.IGNORECASE,
    )
    # Strip unreliable color attributions on screens/peripherals.
    updated = re.sub(
        r"\b(?:a|an|the)\s+(?:charcoal|beige|navy|gray|grey|black|white|brown|maroon|pink|blue|red|green|yellow|orange|purple|cream|silver|gold)\s+"
        r"(tv|monitor|screen|keyboard|mouse|laptop|computer|phone|remote|display)\b",
        r"a \1",
        updated,
        flags=re.IGNORECASE,
    )
    # Normalize robotic activity openers; redundancy removed in dedupe_semantic_facts.
    updated = re.sub(
        r"\bUp close, the (?:action|real work) is\b",
        "The observed activity is",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\bThe activity underway is\b",
        "The observed activity is",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\bThe main activity is\b",
        "The observed activity is",
        updated,
        flags=re.IGNORECASE,
    )
    # Drop banned generic padding sentences outright.
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", updated) if p.strip()]
    filtered: list[str] = []
    for part in parts:
        sentence = part if part.endswith((".", "!", "?")) else part + "."
        if any(pat.match(sentence) for pat in _BANNED_SENTENCE_RES):
            continue
        if re.search(
            r"\b(?:setting remains clearly indoors|share the surrounding space)\b",
            sentence,
            re.IGNORECASE,
        ):
            continue
        filtered.append(sentence)
    updated = " ".join(filtered)
    updated = fix_double_articles(updated)
    updated = drop_malformed_caption_sentences(updated)
    updated = dedupe_object_mention_sentences(updated)
    updated = dedupe_semantic_facts(updated)
    updated = fold_support_object_into_activity(updated)
    updated = fix_double_articles(updated)
    # Subject–verb agreement for plural pronouns (common VLM/Ollama slip).
    updated = re.sub(r"\bthey\s+leads\b", "they lead", updated, flags=re.IGNORECASE)
    updated = re.sub(r"\bthey\s+moves\b", "they move", updated, flags=re.IGNORECASE)
    updated = re.sub(r"\bthey\s+holds\b", "they hold", updated, flags=re.IGNORECASE)
    updated = re.sub(r"\bthey\s+wears\b", "they wear", updated, flags=re.IGNORECASE)
    updated = re.sub(r"\bthey\s+rides\b", "they ride", updated, flags=re.IGNORECASE)
    updated = re.sub(r"\bthey\s+guides\b", "they guide", updated, flags=re.IGNORECASE)
    updated = re.sub(r"\bthey\s+stands\b", "they stand", updated, flags=re.IGNORECASE)
    updated = re.sub(r"\s{2,}", " ", updated).strip()
    # Capitalize the start of each sentence safely; drop empty/orphan fragments.
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", updated) if p.strip()]
    capped: list[str] = []
    for part in parts:
        sentence = part if part.endswith((".", "!", "?")) else part + "."
        # Drop sentences that lost their subject after accessory stripping.
        if re.match(r"^(?:Is|Are|Was|Were)\s+also\s+visible\.?$", sentence, re.IGNORECASE):
            continue
        if re.match(r"^Close beside\b", sentence, re.IGNORECASE):
            continue
        if re.match(r"^In the foreground,?\s+(?:a|an|the)\s+\w+\.?\s*$", sentence, re.IGNORECASE):
            continue
        if len(re.findall(r"[A-Za-z]+", sentence)) < 3:
            continue
        if sentence[0].islower():
            sentence = sentence[0].upper() + sentence[1:]
        capped.append(sentence)
    updated = " ".join(capped)
    updated = order_action_first_sentences(updated)
    updated = re.sub(
        r"\bshare the frame with them\b",
        "are arranged around them",
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(r"\boutdoor\s+outdoors\b", "outdoors", updated, flags=re.IGNORECASE)
    return updated.strip()


def choose_better_caption(*candidates: str) -> str:
    """Pick the highest-scoring non-empty candidate after sanitization."""
    best = ""
    best_score = -1.0
    for raw in candidates:
        cleaned = sanitize_caption(raw)
        if not cleaned:
            continue
        score = caption_sanity_score(cleaned)
        lower = cleaned.lower()
        words = len(cleaned.split())
        # Prefer captions that retain a detected-person subject when present in any candidate.
        if any(
            tok in lower for tok in ("person", "people", "man", "woman", "child", "skier")
        ):
            score += 0.35
        if re.search(
            r"\b(attention stays|softens into the wider|close by are|is close by)\b",
            lower,
        ):
            score -= 0.45
        if re.search(r"\b(\w+)\b.*\band \1\b", lower):
            score -= 0.20
        # Heavily penalize detector-like multi-person summaries.
        if re.search(r"\ba person talking to (?:a |an )?person\b", lower):
            score -= 0.85
        if "second person stands farther back in the frame" in lower and words < 50:
            score -= 0.70
        if re.search(
            r"\b(?:a |an |the )?person(?: wearing [\w\s]+)? "
            r"(?:stands?|is standing|is) (?:near|beside|next to)\b",
            lower,
        ) and words < 28:
            score -= 0.35
        person_hits = len(re.findall(r"\bpersons?\b", lower))
        if person_hits >= 3 and words < 40:
            score -= 0.40
        # Prefer informative natural detail when scores are close.
        score += min(0.22, words / 200.0)
        if 45 <= words <= 160:
            score += 0.08
        # Do not let length alone beat a cleaner factual caption.
        if words > 90:
            score -= 0.05
        if lower.startswith(("there is ", "there are ", "the image shows", "the image depicts")):
            score -= 0.15
        if "farm pasture" in lower:
            score -= 0.10
        if re.search(r"\bobserving the (?:scene|activity|horse)\b", lower):
            score -= 0.85
        if re.search(r"\b(a pair|individuals|one individual)\b", lower):
            score -= 0.45
        if re.search(r"\bkhaki(?:-colored)?\b", lower):
            score -= 0.95
        if re.search(r"\bolive(?:-colored)?\b", lower):
            score -= 0.95
        if re.search(r"\bt-?shirts?\b", lower) and "sweatshirt" not in lower:
            score -= 0.35
        if re.search(r"\bcloser to the camera than\b", lower):
            score -= 0.55
        # Reward multi-entity coverage without requiring checklist style.
        if lower.count("horse") >= 2:
            score += 0.08
        if "fire" in lower:
            score += 0.08
        if score > best_score:
            best_score = score
            best = cleaned
    return best
