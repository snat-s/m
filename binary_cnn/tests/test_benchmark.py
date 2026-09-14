from binary_cnn.benchmark import MIME_LABELS


def test_generic_text_is_not_assigned_to_a_target_language() -> None:
    assert "text/plain" not in MIME_LABELS
    assert MIME_LABELS["text/x-script.python"] == "python"
