def validate_output(output: str) -> tuple[bool, str]:
    """Validates the LLM output. Returns (is_valid, failure_reason)."""
    clean_out = output.strip()
    if not clean_out:
        return False, "The response was completely empty."
    if len(clean_out) < 10:
        return False, "The response was too short to be meaningful."

    refusal_keywords = ["i cannot fulfill", "as an ai", "i am unable to", "against my guidelines"]
    if any(kw in clean_out.lower() for kw in refusal_keywords):
        return False, "The local model safety-refused to execute the prompt."

    return True, ""