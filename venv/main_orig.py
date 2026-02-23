try:
    # Prefer the new package if available (avoids deprecation warnings)
    from langchain_ollama import OllamaLLM as _OllamaLLM
except Exception:
    _OllamaLLM = None

try:
    from langchain_community.llms import Ollama as _OllamaLegacy
except Exception:
    _OllamaLegacy = None
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
try:
    # Prefer core package to avoid importing top-level `langchain` which pulls many
    # optional dependencies and can hang during import in some environments.
    from langchain_core.output_parsers import StrOutputParser
except Exception:
    # Provide a minimal fallback that simply returns the raw string.
    class StrOutputParser:
        def parse(self, text: str) -> str:
            return text
from pydantic import BaseModel
from typing import List

if _OllamaLLM is not None:
    llm = _OllamaLLM(model="llama3")
elif _OllamaLegacy is not None:
    llm = _OllamaLegacy(model="llama3")
else:
    # Provide a simple mock LLM for local testing when no Ollama client is available.
    import json, re

    class MockLLM:
        is_mock = True
        def invoke(self, inputs, **kwargs):
            # inputs may be a dict or string; return JSON strings for parsers.
            if isinstance(inputs, dict):
                if "job_desc" in inputs:
                    jd = inputs["job_desc"]
                    title = jd.split(",")[0].strip()
                    skills = []
                    m = re.search(r"skills\s*(.*?)(?:,|$)", jd, re.IGNORECASE)
                    if m:
                        skills = [s.strip() for s in m.group(1).split() if s.strip()]
                    exp = 0
                    m2 = re.search(r"experience\s*(\d+)", jd, re.IGNORECASE)
                    if m2:
                        exp = int(m2.group(1))
                    job_out = {
                        "job_title": title,
                        "required_skills": skills,
                        "experience_required": exp,
                        "tools": [],
                        "soft_skills": [],
                    }
                    return json.dumps(job_out)
                if "job" in inputs and "resume" in inputs:
                    # Simple resume suggestions
                    out = {
                        "missing_skills": ["SQL"],
                        "improvement_points": ["Add practical projects"],
                        "overall_fit_summary": "Candidate shows potential but needs more hands-on experience.",
                    }
                    return json.dumps(out)
                # fallback
                return json.dumps({"text": str(inputs)})
            return f"MOCK: {inputs}"

    llm = MockLLM()

# Job model
class JobDetails(BaseModel):
    job_title: str
    required_skills: List[str]
    experience_required: int
    tools: List[str]
    soft_skills: List[str]

job_parser = PydanticOutputParser(pydantic_object=JobDetails)

job_prompt = PromptTemplate(
    template="""
Extract structured job details.

{format_instructions}

Job Description:
{job_desc}
""",
    input_variables=["job_desc"],
    partial_variables={"format_instructions": job_parser.get_format_instructions()}
)

# Resume model
class ResumeSuggestions(BaseModel):
    missing_skills: List[str]
    improvement_points: List[str]
    overall_fit_summary: str

resume_parser = PydanticOutputParser(pydantic_object=ResumeSuggestions)

resume_prompt = PromptTemplate(
    template="""
Compare job and resume and give structured suggestions.

{format_instructions}

Job:
{job}

Resume:
{resume}
""",
    input_variables=["job", "resume"],
    partial_variables={"format_instructions": resume_parser.get_format_instructions()}
)

# Cover letter
cover_prompt = PromptTemplate(
    template="""
Write a professional cover letter.

Job:
{job}

Resume:
{resume}
""",
    input_variables=["job", "resume"]
)

# Sample input
job_description = "Python Developer, skills Python ML SQL Git, experience 2 years"
resume_text = "Fresher with Python and ML basics"

# Chains are invoked manually to avoid composing non-runnable parsers with runnables


def _extract_text(result):
    # Normalize various possible LLM return shapes into a plain string
    if isinstance(result, dict):
        text = result.get("text") or result.get("output")
        if not text:
            gens = result.get("generations") or result.get("generation")
            if gens:
                try:
                    first = gens[0]
                    # gens can be nested lists or objects
                    if isinstance(first, list) and len(first) > 0:
                        candidate = first[0]
                    else:
                        candidate = first[0] if isinstance(first, list) else first
                    if isinstance(candidate, dict):
                        text = candidate.get("text") or candidate.get("content")
                    else:
                        text = getattr(candidate, "text", None) or getattr(candidate, "content", None)
                except Exception:
                    text = None
        if text is None:
            text = str(result)
    else:
        text = result
    return text


def _run_template_and_llm(prompt_template, llm_obj, parser_obj, inputs_dict):
    # If using the MockLLM, it expects structured inputs; otherwise render prompt text.
    if getattr(llm_obj, "is_mock", False):
        return llm_obj.invoke(inputs_dict)
    # render prompt text using format_instructions from the parser if available
    fmt = parser_obj.get_format_instructions() if parser_obj is not None else ""
    rendered = prompt_template.format(**inputs_dict, format_instructions=fmt)
    return llm_obj.invoke(rendered)


# Run
raw_job = _run_template_and_llm(job_prompt, llm, job_parser, {"job_desc": job_description})
job_text = _extract_text(raw_job)
job_info = job_parser.parse(job_text)
print("\n===== JOB DETAILS =====")
print(job_info)

raw_resume = _run_template_and_llm(resume_prompt, llm, resume_parser, {"job": job_info, "resume": resume_text})
resume_text_out = _extract_text(raw_resume)
resume_suggestions = resume_parser.parse(resume_text_out)
print("\n===== RESUME SUGGESTIONS =====")
print(resume_suggestions)

raw_cover = _run_template_and_llm(cover_prompt, llm, None, {"job": job_info, "resume": resume_text})
cover_text_out = _extract_text(raw_cover)
cover_letter = StrOutputParser().parse(cover_text_out)
print("\n===== COVER LETTER =====")
print(cover_letter)
