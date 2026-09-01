"""Recording a setting into a user config file without damaging it.

The file these tests exercise is the one that holds the user's API key, so the
cases that matter most are the ones where the writer must refuse.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from cagent.config import CONFIG_FILENAME, load_config, write_setting
from cagent.errors import ConfigError

SECRETS = """\
# My endpoint.
[cagent]
base_url = "https://api.example.com/v1"   # keep this comment
model = "some-model"
api_key = "sk-do-not-lose-me"

# --- sandbox ---
sandbox_mode = "docker"
"""


def _table(path: Path) -> dict[str, object]:
    return tomllib.loads(path.read_text(encoding="utf-8"))["cagent"]


def test_a_new_setting_is_added_without_touching_anything_else(tmp_path: Path) -> None:
    config = tmp_path / CONFIG_FILENAME
    config.write_text(SECRETS, encoding="utf-8")

    write_setting("desktop_path", "D:/repo/desktop", config_file=config)

    table = _table(config)
    assert table["desktop_path"] == "D:/repo/desktop"
    assert table["api_key"] == "sk-do-not-lose-me"
    assert table["base_url"] == "https://api.example.com/v1"
    assert table["sandbox_mode"] == "docker"


def test_the_users_comments_and_ordering_survive(tmp_path: Path) -> None:
    """A serialiser round-trip would silently delete all of this."""
    config = tmp_path / CONFIG_FILENAME
    config.write_text(SECRETS, encoding="utf-8")

    write_setting("desktop_path", "/repo/desktop", config_file=config)

    text = config.read_text(encoding="utf-8")
    assert "# My endpoint." in text
    assert "# keep this comment" in text
    assert "# --- sandbox ---" in text
    assert text.index("base_url") < text.index("sandbox_mode")


def test_writing_twice_replaces_rather_than_duplicates(tmp_path: Path) -> None:
    config = tmp_path / CONFIG_FILENAME
    config.write_text(SECRETS, encoding="utf-8")

    write_setting("desktop_path", "/first", config_file=config)
    write_setting("desktop_path", "/second", config_file=config)

    text = config.read_text(encoding="utf-8")
    assert text.count("desktop_path") == 1
    assert _table(config)["desktop_path"] == "/second"


def test_the_original_is_backed_up_before_being_replaced(tmp_path: Path) -> None:
    config = tmp_path / CONFIG_FILENAME
    config.write_text(SECRETS, encoding="utf-8")

    write_setting("desktop_path", "/repo/desktop", config_file=config)

    backup = tmp_path / f"{CONFIG_FILENAME}.bak"
    assert backup.read_text(encoding="utf-8") == SECRETS


def test_a_missing_file_is_created_with_just_the_table(tmp_path: Path) -> None:
    config = tmp_path / "nested" / CONFIG_FILENAME

    write_setting("desktop_path", "/repo/desktop", config_file=config)

    assert config.read_text(encoding="utf-8") == "[cagent]\ndesktop_path = '/repo/desktop'\n"


def test_a_comment_only_file_gets_a_table_appended(tmp_path: Path) -> None:
    """The shape of a copied example with every setting still switched off."""
    config = tmp_path / CONFIG_FILENAME
    config.write_text("# base_url = \"...\"\n# model = \"...\"\n", encoding="utf-8")

    write_setting("desktop_path", "/repo/desktop", config_file=config)

    assert _table(config)["desktop_path"] == "/repo/desktop"
    assert "# base_url" in config.read_text(encoding="utf-8")


def test_a_commented_out_assignment_is_left_alone_as_documentation(tmp_path: Path) -> None:
    """Only an active assignment is replaced; a hint stays a hint."""
    config = tmp_path / CONFIG_FILENAME
    config.write_text("[cagent]\n# desktop_path = \"/example\"\n", encoding="utf-8")

    write_setting("desktop_path", "/real", config_file=config)

    text = config.read_text(encoding="utf-8")
    assert '# desktop_path = "/example"' in text
    assert _table(config)["desktop_path"] == "/real"


def test_a_windows_path_is_written_as_a_literal_string(tmp_path: Path) -> None:
    r"""Backslashes must survive: a basic string would read ``\v`` as an escape."""
    config = tmp_path / CONFIG_FILENAME

    write_setting("desktop_path", r"D:\vscode\repo\desktop", config_file=config)

    assert _table(config)["desktop_path"] == r"D:\vscode\repo\desktop"


def test_a_value_containing_a_quote_falls_back_to_an_escaped_string(tmp_path: Path) -> None:
    config = tmp_path / CONFIG_FILENAME

    write_setting("desktop_path", "/it's here/desktop", config_file=config)

    assert _table(config)["desktop_path"] == "/it's here/desktop"


def test_settings_from_another_table_are_not_disturbed(tmp_path: Path) -> None:
    config = tmp_path / CONFIG_FILENAME
    config.write_text(f'{SECRETS}\n[other.tool]\nkeep = true\n', encoding="utf-8")

    write_setting("desktop_path", "/repo/desktop", config_file=config)

    document = tomllib.loads(config.read_text(encoding="utf-8"))
    assert document["other"]["tool"]["keep"] is True
    assert document["cagent"]["api_key"] == "sk-do-not-lose-me"


def test_an_unknown_setting_is_refused(tmp_path: Path) -> None:
    config = tmp_path / CONFIG_FILENAME

    with pytest.raises(ConfigError, match="Unknown setting"):
        write_setting("not_a_setting", "x", config_file=config)

    assert not config.exists()


def test_a_broken_file_is_left_exactly_as_it_was(tmp_path: Path) -> None:
    """Never rewrite a file we could not fully understand in the first place."""
    config = tmp_path / CONFIG_FILENAME
    broken = "[cagent]\nbase_url = \n"
    config.write_text(broken, encoding="utf-8")

    with pytest.raises(ConfigError, match="not valid TOML"):
        write_setting("desktop_path", "/repo/desktop", config_file=config)

    assert config.read_text(encoding="utf-8") == broken
    assert not (tmp_path / f"{CONFIG_FILENAME}.bak").exists()


def test_a_file_without_the_cagent_table_is_refused(tmp_path: Path) -> None:
    config = tmp_path / CONFIG_FILENAME
    stray = 'model = "top-level"\n'
    config.write_text(stray, encoding="utf-8")

    with pytest.raises(ConfigError, match=r"\[cagent\] table"):
        write_setting("desktop_path", "/repo/desktop", config_file=config)

    assert config.read_text(encoding="utf-8") == stray


def test_the_result_is_what_the_loader_reads_back(tmp_path: Path) -> None:
    """The point of the exercise: the setting has to actually take effect."""
    config = tmp_path / CONFIG_FILENAME
    config.write_text(SECRETS, encoding="utf-8")

    write_setting("desktop_path", (tmp_path / "desktop").as_posix(), config_file=config)

    loaded = load_config(cwd=tmp_path)
    assert loaded.desktop_path == tmp_path / "desktop"
    assert loaded.api_key == "sk-do-not-lose-me"
