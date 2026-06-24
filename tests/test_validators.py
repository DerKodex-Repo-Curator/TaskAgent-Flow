from agent_flow.validators import validate_output


def test_empty_output_is_invalid() -> None:
    is_valid, reason = validate_output("")
    assert not is_valid
    assert "empty" in reason.lower()


def test_whitespace_only_output_is_invalid() -> None:
    is_valid, _ = validate_output("    \n  ")
    assert not is_valid


def test_none_output_is_invalid() -> None:
    is_valid, _ = validate_output(None)
    assert not is_valid


def test_too_short_output_is_invalid() -> None:
    is_valid, reason = validate_output("ok")
    assert not is_valid
    assert "short" in reason.lower()


def test_refusal_output_is_invalid() -> None:
    is_valid, reason = validate_output("I cannot help with that request.")
    assert not is_valid
    assert "refused" in reason.lower()


def test_reasonable_output_is_valid() -> None:
    is_valid, reason = validate_output("This is a perfectly reasonable response.")
    assert is_valid
    assert reason == ""
