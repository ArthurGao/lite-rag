import json

import pytest
from typer.testing import CliRunner

from lite_rag.cli import app


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://rag:s3cret-pw@localhost:5432/lite_rag",
        "host=localhost port=5432 dbname=lite_rag user=rag password=s3cret-pw",
        "postgresql://rag:s3cret%40pw@localhost/lite_rag",  # '@' inside an encoded password
        "postgresql://s3cret:s3cret@localhost/lite_rag",  # password equal to the user name
    ],
)
def test_config_never_prints_the_database_password(url: str) -> None:
    result = CliRunner().invoke(app, ["config"], env={"LITERAG_DATABASE_URL": url})
    assert result.exit_code == 0, result.output
    shown = json.loads(result.output)["database_url"]
    assert "password=***" in shown and "localhost" in shown
    assert "password=s3cret" not in shown and "s3cret-pw" not in shown


def test_an_unparseable_url_is_replaced_not_echoed() -> None:
    result = CliRunner().invoke(app, ["config"], env={"LITERAG_DATABASE_URL": "pw=s3cret ??"})
    assert result.exit_code == 0, result.output
    assert "s3cret" not in result.output
