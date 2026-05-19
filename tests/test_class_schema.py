from training.class_schema import CORE_CLASSES, EXTENDED_CLASSES, get_schema, schema_name


def test_get_schema_core():
    assert get_schema("core") == CORE_CLASSES
    assert get_schema("4") == CORE_CLASSES


def test_get_schema_extended():
    assert get_schema("extended") == EXTENDED_CLASSES
    assert get_schema("8classes") == EXTENDED_CLASSES


def test_schema_name_roundtrip():
    assert schema_name(CORE_CLASSES) == "core"
    assert schema_name(EXTENDED_CLASSES) == "extended"
