from __future__ import annotations


IT_TRANSLATION_CONTRACT = " ".join([
    "Translation context: software engineering, AI systems, security/privacy, and GitHub documentation for Chinese developers and vibe coders.",
    "Translate the intended technical meaning in its sentence and project context, never word by word. Preserve product names, commands, API names, and code.",
    "Use natural concise Chinese that a Chinese developer would actually say. Prefer established technical collocations over literal dictionary meanings.",
    "Resolve polysemy from the technical context rather than a general-purpose dictionary. For names and institutions, use an established Chinese form only when certain; otherwise preserve the source spelling.",
    "Read the surrounding README paragraph before translating an isolated highlighted phrase.",
])


PLAIN_CHINESE_CONTRACT = " ".join([
    "Viewer-facing Chinese must sound like one Chinese developer explaining a concrete event to another, not like a research report, consultant memo, or translated press release.",
    "Every screen should answer in plain words: who did what, using which concrete product/model when known, and what happened next.",
    "Prefer verbs and objects over relationship labels and abstract noun clusters. Translate technical nouns into the concrete action they describe when that is clearer.",
    "Use short spoken clauses. A viewer should understand each line on the first read without knowing the English source.",
])
