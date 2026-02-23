# 🚀 AI Job Application Assistant

AI Job Application Assistant is an intelligent AI-powered system that analyzes job descriptions, extracts structured job requirements, evaluates resume skill gaps, computes skill match scores, and generates personalized professional cover letters using Large Language Models (LLMs).

This project demonstrates **enterprise-grade structured LLM pipelines, schema validation, and prompt engineering**, suitable for HR automation workflows.

---

## 🎯 Objective

To build an AI assistant that helps candidates tailor their job applications by automatically analyzing job descriptions and resumes using structured AI outputs.

---

## 🧠 Key Features

### ✅ Job Description Analyzer
- Extracts structured job information such as:
  - Job Title  
  - Required Skills  
  - Experience Required  
  - Tools  
  - Soft Skills  
- Uses **PydanticOutputParser** for schema validation.

### ✅ Resume Skill Gap Analyzer
- Identifies missing skills
- Provides improvement suggestions
- Summarizes candidate-job fit

### ✅ Skill Match Score
- Calculates percentage match between resume skills and job requirements.

### ✅ AI Cover Letter Generator
- Generates a professional, customized cover letter using LLMs.
- Uses **StrOutputParser** for natural language output.

### ✅ Mock LLM for Offline Testing
- Allows project execution without Ollama or internet dependency.
- Ensures reproducibility and portability.

### ✅ Enterprise Logging
- Implements logging for debugging and monitoring pipeline execution.

---

## 🏗️ Architecture Pipeline
