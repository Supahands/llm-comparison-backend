
def get_question_generation_system_prompt(required_tag_count: int) -> str:
    """Returns the system prompt for question generation with specified tag count."""
    return f"""You are a precise AI assistant that creates clear, focused questions.
When generating questions with tags:
1. ALWAYS use EXACTLY {required_tag_count} tag(s) per question - no more, no fewer
2. Create concise questions that are only as detailed as necessary
3. Only use paragraph-length when complexity truly requires it
4. Format your response as a valid JSON object with the structure requested
5. Each question should be direct and to the point
6. Tags can be simple words OR short phrases (up to 5 words maximum)"""


TAG_REQUIREMENTS = """
Tag Requirements:
- EVERY question MUST have EXACTLY the specified number of tags
- Tags should be clear, relevant, and concise
- Tags can be single words OR short phrases (maximum 5 words)
- First tag should represent the primary capability/category
- Additional tags should add specificity

Question Quality Requirements:
- Questions should be concise yet detailed enough to be clear
- Only use paragraph-length when the complexity truly requires it
- Include necessary context, but be as brief as possible
- Provide clear instructions on what's expected
- Focus on topics where AI can demonstrate reasoning, creativity, or knowledge
- Avoid unnecessary verbosity or filler text
"""


def get_basic_question_prompt() -> str:
    """Returns the prompt for generating questions without input question."""
    return """Generate 4 diverse questions for evaluating language models.

Required JSON Schema:
{
    "questions": [
        {
            "question": "Concise question text that's only as detailed as necessary",
            "tags": ["capability"] // EXACTLY ONE TAG - NO EXCEPTIONS
        },
        ... // 3 more questions, each with EXACTLY ONE TAG
    ]
}

CRITICAL REQUIREMENTS:
1. Each question MUST have EXACTLY ONE TAG - no exceptions
2. Questions should be concise yet detailed enough to be clear
3. Only use paragraph-length for complex topics that require it
4. Keep questions focused and to the point
5. Create questions that test different cognitive abilities:
   - One testing analytical reasoning
   - One testing creative thinking
   - One testing knowledge application
   - One testing logical problem solving
6. NO questions requiring very recent information or real-time data

DOUBLE CHECK: I need EXACTLY 4 questions, each with EXACTLY 1 tag.
"""


def get_untagged_question_prompt(input_question: str, tag_requirements: str) -> str:
    """Returns the prompt for questions with input but no tags."""
    return f"""Analyze this question and generate 4 related questions.

Input Question: "{input_question}"

Required JSON Schema:
{{
    "questions": [
        {{
            "question": "{input_question}",
            "tags": ["primaryCapability"]
        }},
        ... // 3 more related questions, each with EXACTLY ONE TAG
    ]
}}

CRITICAL REQUIREMENTS:
1. First question must be the input question with one appropriate tag
2. Generate 3 additional related questions that explore similar themes or concepts
3. Each question MUST have EXACTLY ONE TAG - no exceptions
4. Keep questions concise yet clear and focused
5. Only use paragraph-length if absolutely necessary for clarity
6. Questions should be direct and to the point

{tag_requirements}"""


def get_tagged_question_prompt(input_question: str, original_tags: list, new_tag_count: int) -> str:
    """Returns the prompt for questions with input and existing tags."""
    tag_list = ", ".join([f'"{tag}"' for tag in original_tags])
    
    return f"""Analyze this tagged question and generate 4 related questions.

Input Question: {{
    "question": "{input_question}",
    "tags": [{tag_list}]
}}

Required JSON Schema:
{{
    "questions": [
        {{
            "question": "Concise question that provides necessary context and clear instructions",
            "tags": [{tag_list}, "additionalTag"] // EXACTLY {new_tag_count} tags
        }},
        ... // 3 more questions, each with EXACTLY {new_tag_count} tags
    ]
}}

CRITICAL REQUIREMENTS:
1. EVERY question MUST have EXACTLY {new_tag_count} tags total
2. Keep ALL original tags: [{tag_list}] in the EXACT SAME order
3. Add EXACTLY ONE NEW relevant tag at the end (word or short phrase up to 5 words)
4. Make questions concise yet clear and detailed:
   - Only as long as necessary to provide context and instructions
   - Use paragraph-length ONLY when complexity truly requires it
   - Prefer shorter, more focused questions when possible
5. Questions must be related to the input theme but explore different aspects

DOUBLE-CHECK: Count the tags for each question - there should be EXACTLY {new_tag_count} tags per question."""
