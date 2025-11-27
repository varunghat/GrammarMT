if prompt_name == "prompt_basic_1":
    variety_part = f" ({variety_name} variety)" if include_variety else ""
    source_part = "Romansh " if include_source_language else ""
    return f"Translate the following {source_part}segment surrounded in triple backticks into German. The {source_part}{variety_part} segment: ```{romansh_sentence}``` Only give me the translation in German in plain text."
try:
    if enable_reasoning:
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=api_messages
        )
    else:
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=api_messages,
            extra_body={
                "thinking": {"type": "disabled", "budget_tokens": 0}
            }
        )