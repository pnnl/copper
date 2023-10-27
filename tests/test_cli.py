import click, logging
from click.testing import CliRunner
from copper.cli import run


def test_cli_incorrect_file():
    runner = CliRunner()
    result = runner.invoke(run, ["test"])
    assert "'test': No such file or directory" in result.output
    assert result.exit_code == 2


def test_cli_correct(caplog):
    caplog.set_level(logging.INFO)
    runner = CliRunner()
    result = runner.invoke(run, ["./tests/data/cli_input_file.json"])
    assert result.exit_code == 0
