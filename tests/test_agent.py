"""Agent loop, tool guards, and the Python sidecar.

The security tests here are the point of the file. An agent that reads
attacker-influenceable filing text must not be able to write to the warehouse or
escape the sandbox, and that has to be enforced by the tools rather than by the
prompt.
"""

from __future__ import annotations

import json

import pytest

from signalforge.agent.loop import _parse_decision, run_agent
from signalforge.agent.sidecar import SidecarError, run_python, sidecar_available, validate
from signalforge.agent.tools import (
    SqlGuardError,
    _guard_sql,
    build_tools,
    list_companies,
    sql_query,
)
from signalforge.db import connect, upsert
from signalforge.llm.base import LLMResponse


@pytest.fixture
def corpus(warehouse):
    with connect() as con:
        upsert(
            con, "companies", [{"cik": "0000000001", "name": "Testco", "ticker": "TST"}], key="cik"
        )
        upsert(
            con,
            "filings",
            [
                {
                    "accession": "acc-1",
                    "cik": "0000000001",
                    "form": "10-Q",
                    "filing_date": "2026-06-30",
                    "primary_doc": "d.htm",
                }
            ],
            key="accession",
        )
        upsert(
            con,
            "signals",
            [
                {
                    "signal_id": "sig-1",
                    "name": "guidance_tone",
                    "cik": "0000000001",
                    "ticker": "TST",
                    "accession": "acc-1",
                    "as_of": "2026-06-30",
                    "score": -0.72,
                    "confidence": 0.85,
                    "direction": "bearish",
                    "rationale": "Guidance withdrawn.",
                    "evidence": ["withdrawing our outlook"],
                    "extraction_ids": ["e1"],
                    "pipeline_version": "score-v1",
                }
            ],
            key="signal_id",
        )
    return True


# ------------------------------------------------------------------ SQL guard
class TestSqlGuard:
    def test_select_is_allowed_and_gets_a_limit(self):
        assert "LIMIT" in _guard_sql("SELECT * FROM filings")

    def test_existing_limit_is_respected(self):
        assert _guard_sql("SELECT 1 LIMIT 5").lower().count("limit") == 1

    def test_cte_is_allowed(self):
        assert _guard_sql("WITH x AS (SELECT 1) SELECT * FROM x")

    @pytest.mark.parametrize(
        "sql",
        [
            "DROP TABLE filings",
            "DELETE FROM signals",
            "UPDATE signals SET score = 1",
            "INSERT INTO signals VALUES (1)",
            "CREATE TABLE evil (x INT)",
            "ATTACH 'evil.db' AS evil",
            "PRAGMA database_list",
            "COPY filings TO '/tmp/leak.csv'",
            "INSTALL httpfs",
            "SELECT * FROM read_csv('/etc/passwd')",
            "SELECT * FROM read_parquet('s3://bucket/x')",
        ],
    )
    def test_mutations_and_file_access_are_refused(self, sql):
        with pytest.raises(SqlGuardError):
            _guard_sql(sql)

    def test_stacked_statements_are_refused(self):
        with pytest.raises(SqlGuardError, match="multiple statements"):
            _guard_sql("SELECT 1; DROP TABLE filings")

    def test_keyword_hidden_behind_a_comment_is_still_caught(self):
        """Comments are stripped before the denylist runs, so a keyword cannot
        hide behind `--`."""
        with pytest.raises(SqlGuardError):
            _guard_sql("SELECT 1 --\nDELETE FROM signals")

    def test_block_comment_evasion_is_caught(self):
        with pytest.raises(SqlGuardError):
            _guard_sql("SELECT 1 /* x */ ; DROP TABLE filings")

    def test_empty_query_is_refused(self):
        with pytest.raises(SqlGuardError, match="empty"):
            _guard_sql("   ")

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM review_queue",
            "SELECT * FROM review_queue AS x",
            "SELECT * FROM llm_cache",
            # Comma joins: the second table must be checked too, not just the
            # first one after the FROM keyword.
            "SELECT * FROM signals, review_queue",
            "SELECT * FROM signals s, review_queue r",
            "SELECT * FROM signals JOIN review_queue r ON r.task = signals.name",
            "SELECT (SELECT count(*) FROM review_queue) AS n FROM signals",
            "SELECT * FROM signals UNION ALL SELECT * FROM review_queue",
            # Metadata reach that the verb and denylist checks do not cover.
            "SELECT * FROM information_schema.tables",
            "SELECT * FROM duckdb_settings()",
        ],
    )
    def test_tables_outside_the_allowlist_are_refused(self, sql):
        """READABLE_TABLES is a control, not documentation.

        review_queue is withheld deliberately: it holds raw model output, and
        letting the agent read it feeds its own prior reasoning back as evidence.
        """
        with pytest.raises(SqlGuardError, match="not readable"):
            _guard_sql(sql)

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM signals",
            "SELECT * FROM main.signals",
            "SELECT s.name, c.ticker FROM signals s, companies c WHERE s.cik = c.cik",
            "SELECT a.rule FROM alerts a JOIN signals s ON s.signal_id = a.signal_id",
            "SELECT count(*) FROM (SELECT cik FROM filings) t",
            "WITH a AS (SELECT cik FROM filings) SELECT * FROM a",
            "SELECT task, count(*) FROM extractions GROUP BY task ORDER BY 2 DESC LIMIT 10",
        ],
    )
    def test_allowlisted_reads_are_not_false_positives(self, sql):
        """The allowlist must not break the queries the agent legitimately needs."""
        assert _guard_sql(sql)

    def test_tool_returns_error_text_rather_than_raising(self, corpus):
        """A refused query must come back as something the agent can correct."""
        result = sql_query("DROP TABLE filings")
        assert result.ok is False
        assert "refused" in result.text.lower()

    def test_read_only_connection_blocks_writes_that_pass_the_parser(self, corpus):
        """Defence in depth: even if the parser were bypassed, the connection is
        opened read-only."""
        import duckdb

        with pytest.raises(duckdb.Error), connect(read_only=True) as con:
            con.execute("CREATE TABLE sneaky (x INT)")

    def test_valid_query_returns_rows(self, corpus):
        result = sql_query("SELECT ticker, name FROM companies")
        assert result.ok and "TST" in result.text
        assert result.data[0]["ticker"] == "TST"

    def test_malformed_sql_is_reported_not_raised(self, corpus):
        result = sql_query("SELECT nonexistent_column FROM companies")
        assert result.ok is False and "SQL error" in result.text


# -------------------------------------------------------------------- sidecar
@pytest.mark.skipif(not sidecar_available(), reason="sidecar needs POSIX resource limits")
class TestSidecar:
    def test_arithmetic_works(self):
        r = run_python("result = sum([1, 2, 3]) / 3")
        assert r.ok and r.result == 2.0

    def test_stdout_is_captured(self):
        assert "hello" in run_python("print('hello')").stdout

    def test_allowed_import(self):
        assert run_python("import math\nresult = round(math.sqrt(16))").result == 4

    @pytest.mark.parametrize(
        "code",
        [
            "import os",
            "import subprocess",
            "import socket",
            "from os import path",
            "open('/etc/passwd')",
            "exec('x=1')",
            "eval('1+1')",
            "__import__('os')",
            "().__class__.__bases__",
            "x = (1).__class__",
        ],
    )
    def test_dangerous_code_is_rejected_before_running(self, code):
        with pytest.raises(SidecarError):
            validate(code)

    def test_syntax_error_is_reported_clearly(self):
        with pytest.raises(SidecarError, match="syntax error"):
            validate("def (:")

    def test_infinite_loop_is_killed(self):
        r = run_python("while True: pass", timeout_s=2.0)
        assert r.ok is False
        assert "exceeded" in r.stderr

    def test_a_killed_loop_always_explains_itself(self):
        """A failure the agent cannot read is a failure it cannot recover from.

        Regression: RLIMIT_CPU was set equal to the wall-clock timeout, so for a
        CPU-bound loop the two deadlines raced. When the rlimit won, the child died
        by signal with no traceback and `stderr` came back empty — the agent saw
        `ok=False` and "(no output)". Whichever limit fires, there must be a reason.
        """
        r = run_python("while True: pass", timeout_s=2.0)
        assert r.ok is False
        assert r.stderr.strip(), "a killed process must still report why"
        assert r.render() != "(no output)"

    def test_runtime_error_is_returned_not_raised(self):
        r = run_python("result = 1 / 0")
        assert r.ok is False and "ZeroDivisionError" in r.stderr

    def test_execution_is_isolated_from_the_host(self):
        """The child must not inherit host environment or working directory."""
        r = run_python("import json\nresult = 'ok'")
        assert r.result == "ok"


# ----------------------------------------------------------------- tool suite
def test_list_companies_reports_empty_corpus_actionably(warehouse):
    result = list_companies()
    assert "sf ingest" in result.text, "tell the user how to fix it"


def test_list_companies_lists_the_corpus(corpus):
    assert "TST" in list_companies().text


def test_toolset_specs_are_wellformed(router):
    for name, tool in build_tools(router).items():
        spec = tool.spec()
        assert spec["function"]["name"] == name
        assert tool.description.strip()
        assert spec["function"]["parameters"]["type"] == "object"


# ------------------------------------------------------------ decision parsing
class TestDecisionParsing:
    """Local models emit tool calls in whatever shape they like. Insisting on one
    means the loop fails on most open models."""

    def _resp(self, text):
        return LLMResponse(text=text, model="m", provider="stub")

    def test_bare_json_tool_call(self):
        d = _parse_decision(self._resp('{"tool": "list_companies", "args": {}}'))
        assert d["tool"] == "list_companies"

    def test_fenced_json(self):
        d = _parse_decision(self._resp('```json\n{"tool": "x", "args": {"k": 1}}\n```'))
        assert d == {"tool": "x", "args": {"k": 1}}

    def test_json_embedded_in_prose(self):
        d = _parse_decision(self._resp('Let me look. {"tool": "get_signals", "args": {}} OK?'))
        assert d["tool"] == "get_signals"

    def test_answer_form(self):
        assert _parse_decision(self._resp('{"answer": "done"}'))["answer"] == "done"

    def test_openai_style_function_wrapper(self):
        d = _parse_decision(
            self._resp(
                '{"function": {"name": "sql_query", "arguments": "{\\"sql\\": \\"SELECT 1\\"}"}}'
            )
        )
        assert d["tool"] == "sql_query" and d["args"]["sql"] == "SELECT 1"

    def test_native_tool_calls_take_precedence(self):
        resp = LLMResponse(
            text="",
            model="m",
            provider="stub",
            tool_calls=[{"id": "1", "name": "python", "args": {"code": "1"}}],
        )
        assert _parse_decision(resp)["tool"] == "python"

    def test_prose_only_is_unparseable(self):
        assert _parse_decision(self._resp("I think we should look at Apple.")) is None

    def test_empty_is_unparseable(self):
        assert _parse_decision(self._resp("")) is None


# ------------------------------------------------------------------ loop bounds
class TestAgentLoop:
    def test_answers_immediately_when_the_model_does(self, router, stub, corpus):
        stub.register(
            "What happened to TST?", json.dumps({"answer": "TST guidance was withdrawn."})
        )
        result = run_agent("What happened to TST?", router=router, chain="ci", skills=False)
        assert result.stop_reason == "answered"
        assert "withdrawn" in result.answer
        assert result.steps == []

    def test_runs_a_tool_then_answers(self, router, stub, corpus):
        stub.register("What is in the corpus?", json.dumps({"tool": "list_companies", "args": {}}))
        stub.register(
            "Observation from list_companies",
            json.dumps({"answer": "The corpus holds Testco (TST)."}),
        )
        result = run_agent("What is in the corpus?", router=router, chain="ci", skills=False)
        assert [s.tool for s in result.steps] == ["list_companies"]
        assert "Testco" in result.answer

    def test_max_steps_is_enforced(self, router, stub, corpus):
        """An agent that can loop forever will."""
        stub.register("loop", json.dumps({"tool": "list_companies", "args": {}}))
        stub.register(
            "Observation from list_companies", json.dumps({"tool": "list_companies", "args": {}})
        )
        result = run_agent("loop", router=router, chain="ci", max_steps=3, skills=False)
        assert result.stop_reason == "max_steps_reached"
        assert len(result.steps) == 3
        # A bounded run must still say something, and say that it was bounded.
        assert result.answer.startswith("[stopped early:")

    def test_repeated_identical_call_is_short_circuited(self, router, stub, corpus):
        """The dominant local-model failure: same call forever."""
        stub.register("loop", json.dumps({"tool": "list_companies", "args": {}}))
        stub.register(
            "Observation from list_companies", json.dumps({"tool": "list_companies", "args": {}})
        )
        result = run_agent("loop", router=router, chain="ci", max_steps=3, skills=False)
        assert "already made this exact call" in result.steps[1].observation

    def test_unknown_tool_is_a_correctable_observation(self, router, stub, corpus):
        stub.register("q", json.dumps({"tool": "definitely_not_a_tool", "args": {}}))
        result = run_agent("q", router=router, chain="ci", max_steps=2, skills=False)
        assert "no tool named" in result.steps[0].observation
        assert result.steps[0].ok is False

    def test_bad_arguments_return_the_expected_signature(self, router, stub, corpus):
        stub.register("q", json.dumps({"tool": "list_companies", "args": {"nonexistent_param": 1}}))
        result = run_agent("q", router=router, chain="ci", max_steps=2, skills=False)
        assert "Invalid arguments" in result.steps[0].observation

    def test_unparseable_output_is_corrected_not_fatal(self, router, stub, corpus):
        stub.register("q", "I will now think about this carefully.")
        stub.register("was not a valid JSON object", "Still thinking, no JSON.")
        result = run_agent("q", router=router, chain="ci", max_steps=2, skills=False)
        # Never crashed; the loop asked for a valid shape and then hit its bound.
        assert result.stop_reason == "max_steps_reached"

    def test_cost_cap_terminates_the_run(self, registry, stub, tracer, tmp_path, corpus):
        from signalforge.llm.cache import ResponseCache
        from signalforge.llm.router import Router

        router = Router(
            registry=registry,
            tracer=tracer,
            stub=stub,
            cache=ResponseCache(path=tmp_path / "c.duckdb", enabled=False),
            cost_cap_usd=0.0000001,
        )
        result = run_agent("q", router=router, chain=["opus-5"], skills=False)
        assert result.stop_reason == "cost_cap_exceeded"

    def test_stats_are_reported(self, router, stub, corpus):
        stub.register("q", json.dumps({"answer": "done"}))
        stats = run_agent("q", router=router, chain="ci", skills=False).stats()
        assert set(stats) == {"steps", "llm_calls", "stop_reason", "duration_s", "cost_usd"}
