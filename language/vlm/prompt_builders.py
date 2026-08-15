"""Model-specific prompt builders for deep, evidence-grounded visual narration."""

from __future__ import annotations

from core.contracts.reasoning import SceneUnderstanding


_EVIDENCE_RULES = (
    "RULES (mandatory):\n"
    "1. Describe what you see in simple, everyday language — like explaining the photo to a friend.\n"
    "2. Use supported facts from the evidence package; never contradict them.\n"
    "3. NEVER invent identities, ages, professions, emotions, brands, or unseen objects.\n"
    "4. Prefer observation over interpretation; omit uncertain details.\n"
    "5. Cover main subjects, actions, spatial relationships, important objects, colors when clear, "
    "background/secondary subjects, environment, and readable text when evidence supports them.\n"
    "6. Do NOT write detector-style lines such as 'A person talking to a person' or "
    "'A second person stands farther back in the frame' as the whole caption.\n"
    "7. Do NOT pad with filler. Every sentence must add meaningful visual information.\n"
    "8. Avoid stiff openers like 'The image depicts', 'The image shows', or 'The scene illustrates'.\n"
    "9. Write ONE coherent natural paragraph (not a checklist). For a detailed scene, aim for "
    "about 80–140 words with several connected sentences; for a simple scene, stay concise.\n"
    "10. Connect related details in flowing prose — never emit one short sentence per object.\n"
    "11. No lists, JSON, IDs, or labels like person_1 / Object 2.\n"
)


class VisionPromptBuilder:
    """Build optimized prompts per adapter family — never reuse one generic template."""

    def describe_prompt(self, adapter_name: str) -> str:
        if adapter_name.startswith("florence"):
            return "<MORE_DETAILED_CAPTION>"
        if adapter_name == "qwen":
            return (
                "Observe this image carefully as a professional visual analyst. "
                "In one natural paragraph, describe people, clothing categories and colors, "
                "hairstyles, accessories, actions, object relationships, lighting, and environment. "
                "Prefer precise color names (navy blue, charcoal, burgundy) when clear. "
                "Do not invent details. Do not use bullet points or lists."
            )
        if adapter_name == "internvl":
            return (
                "Provide a single coherent paragraph of scene understanding. "
                "Emphasize accurate clothing recognition, natural color names, spatial relationships, "
                "and activities supported by visual evidence. Avoid speculation and lists."
            )
        if adapter_name == "blip":
            return "a richly detailed photograph of"
        return (
            "Describe the image in one fluent, information-rich paragraph covering people, "
            "clothing and colors when clear, actions, relationships, important objects, "
            "background, and environment. Do not invent unsupported details."
        )

    def narrate_prompt(self, adapter_name: str, understanding: SceneUnderstanding) -> str:
        brief = understanding.evidence_brief.strip()
        if adapter_name.startswith("florence"):
            # Florence task tokens must be alone — evidence fusion happens in NaturalCaptionService.
            return "<MORE_DETAILED_CAPTION>"
        if adapter_name == "qwen":
            return (
                "You are a competition-grade multimodal vision assistant. "
                "You receive BOTH the image and a structured high-confidence evidence package. "
                "Write a professional, detailed, natural description of the visible scene. "
                "Prioritize concrete visual evidence: subjects, actions, spatial relationships, "
                "important objects, environment, clothing/colors when clear, and OCR when present.\n\n"
                f"{_EVIDENCE_RULES}\n"
                f"Verified semantic evidence package:\n{brief}\n\n"
                "Paragraph:"
            )
        if adapter_name == "internvl":
            return (
                "Produce a detailed natural-language description of the image. "
                "Ground every claim in the evidence package; use the image for fluency and layout. "
                f"{_EVIDENCE_RULES}\n"
                f"Evidence package:\n{brief}\n\n"
                "Description:"
            )
        if adapter_name == "blip":
            seed = brief.replace("\n", " ")[:400]
            return (
                f"a richly detailed photo matching this verified evidence without inventing "
                f"extra attributes: {seed}"
            )
        return (
            "Look at the image carefully and write a fluent, detailed observational paragraph. "
            "Ground claims in the evidence package; use the image for spatial layout and appearance. "
            f"{_EVIDENCE_RULES}\n"
            f"Evidence package:\n{brief}\n\n"
            "Paragraph:"
        )

    def alternate_narrate_prompt(self, adapter_name: str, understanding: SceneUnderstanding) -> str:
        """Second-style prompt for candidate diversity (still evidence-grounded)."""
        brief = understanding.evidence_brief.strip()
        if adapter_name.startswith("florence"):
            return self.narrate_prompt(adapter_name, understanding)
        if adapter_name == "qwen":
            return (
                "Rewrite a careful visual observation as one warm, fluent paragraph. "
                "Lead with people, clothing, and colors from the evidence package; then relationships "
                "and place. Prefer evidence over visual guesses. No lists.\n\n"
                f"{_EVIDENCE_RULES}\n"
                f"Evidence package:\n{brief}\n\nParagraph:"
            )
        if adapter_name == "internvl":
            return (
                "Describe the image in one elegant paragraph emphasizing clothing, color, "
                "and spatial relationships supported by the evidence package.\n\n"
                f"{_EVIDENCE_RULES}\n"
                f"Evidence package:\n{brief}\n\nParagraph:"
            )
        return self.narrate_prompt(adapter_name, understanding)
