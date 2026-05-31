from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperLensSkill:
    name: str
    responsibility: str
    consumes: tuple[str, ...]
    produces: tuple[str, ...]
    mutates_memory: bool = False


PAPERLENS_CORE_SKILLS: tuple[PaperLensSkill, ...] = (
    PaperLensSkill(
        name="ReaderSkill",
        responsibility="Read a page or section window and emit durable MemoryPatch operations.",
        consumes=("paper_map", "page_text", "page_visuals", "PaperMemoryV3"),
        produces=("MemoryPatchSet",),
        mutates_memory=True,
    ),
    PaperLensSkill(
        name="EvidenceSkill",
        responsibility="Ground claims in page text, captions, figures, and tables.",
        consumes=("PaperMemoryV3.claims", "paper.search_text", "paper.find_figures"),
        produces=("EvidenceRef", "claim_evidence_links"),
        mutates_memory=True,
    ),
    PaperLensSkill(
        name="CriticSkill",
        responsibility="Find missing contributions, unsupported claims, overclaims, and evaluation gaps.",
        consumes=("PaperMemoryV3", "local_tool_observations"),
        produces=("MemoryCriticAudit", "RereadTask"),
    ),
    PaperLensSkill(
        name="RepairSkill",
        responsibility="Apply targeted reread results as MemoryPatch operations.",
        consumes=("PaperMemoryV3", "MemoryCriticAudit", "targeted_pages"),
        produces=("MemoryPatchSet",),
        mutates_memory=True,
    ),
    PaperLensSkill(
        name="ReportComposerSkill",
        responsibility="Render PaperMemory through ReportPlan, section drafts, section audits, and assembly.",
        consumes=("PaperMemoryV3", "local_tool_observations"),
        produces=("ReportPlan", "ReportSection", "ReportSectionAudit", "PaperReport"),
    ),
    PaperLensSkill(
        name="QASkill",
        responsibility="Answer questions from PaperMemory, library memory, and focused paper evidence.",
        consumes=("question", "PaperMemoryV3", "library_record", "local_tool_observations"),
        produces=("QAAnswer", "QATrace"),
    ),
)


def skill_names() -> list[str]:
    return [skill.name for skill in PAPERLENS_CORE_SKILLS]
