# cv-parser

# Overview

**Value**: Allows the user to extract key information from resumes/CVs in *pdf* and *png* formats.

**Approach**:

Two approaches are tested:

1) A simple approach using regex to extract: name, email and phone number

2) A more complex approach using a QA model from Hugging Face to extract: name, email and phone number, univeristy, degree type, degree course, job title


# To use

Run the `main.py` script

# Future ideas
- Extract text via pre-defined lists & keywords, NER, fine-tune models, mix of techniques, more advanced LLMs
- Look into extracting multiple job titles, job synopsis, hobbies, skills etc
