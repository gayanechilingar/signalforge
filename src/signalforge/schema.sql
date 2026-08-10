-- SignalForge warehouse schema.
--
-- One file, idempotent, applied on every connection open. At this scale that is
-- simpler and safer than a migration tool, and it means a fresh clone is one
-- command from working. See docs/adr/0002-duckdb-warehouse.md for why DuckDB.

-- ---------------------------------------------------------------- raw corpus
CREATE TABLE IF NOT EXISTS companies (
    cik            VARCHAR PRIMARY KEY,   -- zero-padded 10-digit
    ticker         VARCHAR,
    name           VARCHAR NOT NULL,
    sic            VARCHAR,
    sic_description VARCHAR,
    exchange       VARCHAR,
    fetched_at     TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS filings (
    accession      VARCHAR PRIMARY KEY,   -- 0000320193-26-000020
    cik            VARCHAR NOT NULL,
    form           VARCHAR NOT NULL,      -- 10-K | 10-Q | 8-K
    filing_date    DATE NOT NULL,
    report_date    DATE,
    items          VARCHAR,               -- 8-K item codes, comma separated
    primary_doc    VARCHAR,
    url            VARCHAR,
    size_bytes     BIGINT,
    fetched_at     TIMESTAMP DEFAULT current_timestamp
);

-- Extracted plain text of a filing, split into logical sections
-- (Item 1A Risk Factors, Item 7 MD&A, ...). Section-aware because a whole 10-K
-- blows any context window and because signals are section-specific.
CREATE TABLE IF NOT EXISTS sections (
    section_id     VARCHAR PRIMARY KEY,   -- {accession}:{slug}
    accession      VARCHAR NOT NULL,
    slug           VARCHAR NOT NULL,      -- risk_factors | mdna | item_5_02 | body
    heading        VARCHAR,
    ordinal        INTEGER,
    char_len       INTEGER,
    text           VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id       VARCHAR PRIMARY KEY,   -- {section_id}:{ordinal}
    section_id     VARCHAR NOT NULL,
    accession      VARCHAR NOT NULL,
    cik            VARCHAR NOT NULL,
    ordinal        INTEGER NOT NULL,
    token_estimate INTEGER,
    text           VARCHAR NOT NULL
);

-- Embeddings live in their own table so re-embedding with a different model is
-- additive rather than destructive, and so an embedding-model change is
-- traceable in evals.
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id       VARCHAR NOT NULL,
    model          VARCHAR NOT NULL,
    dim            INTEGER NOT NULL,
    vec            FLOAT[] NOT NULL,
    created_at     TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (chunk_id, model)
);

-- ------------------------------------------------------------- LLM outputs
-- Every extraction records exactly what produced it: prompt version + content
-- hash, model, config hash, cost, latency. This is the reproducibility
-- backbone — an extraction you cannot attribute is an extraction you cannot
-- trust or regress against.
CREATE TABLE IF NOT EXISTS extractions (
    extraction_id  VARCHAR PRIMARY KEY,
    task           VARCHAR NOT NULL,
    accession      VARCHAR NOT NULL,
    cik            VARCHAR NOT NULL,
    section_id     VARCHAR,
    prompt_name    VARCHAR NOT NULL,
    prompt_version VARCHAR NOT NULL,
    prompt_hash    VARCHAR NOT NULL,
    model          VARCHAR NOT NULL,
    provider       VARCHAR NOT NULL,
    config_hash    VARCHAR,
    payload        JSON NOT NULL,
    valid          BOOLEAN NOT NULL,
    repair_attempts INTEGER DEFAULT 0,
    grounded_ratio DOUBLE,                -- share of quoted evidence found in source
    tokens_in      INTEGER,
    tokens_out     INTEGER,
    cost_usd       DOUBLE,
    latency_ms     DOUBLE,
    cached         BOOLEAN DEFAULT FALSE,
    error          VARCHAR,
    created_at     TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS signals (
    signal_id      VARCHAR PRIMARY KEY,
    name           VARCHAR NOT NULL,      -- e.g. risk_delta
    cik            VARCHAR NOT NULL,
    ticker         VARCHAR,
    accession      VARCHAR NOT NULL,
    as_of          DATE NOT NULL,
    score          DOUBLE NOT NULL,       -- normalised to [-1, 1]
    confidence     DOUBLE,
    direction      VARCHAR,               -- bullish | bearish | neutral
    rationale      VARCHAR,
    evidence       JSON,
    extraction_ids JSON,
    pipeline_version VARCHAR,
    created_at     TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id       VARCHAR PRIMARY KEY,
    signal_id      VARCHAR NOT NULL,
    rule           VARCHAR NOT NULL,
    severity       VARCHAR NOT NULL,      -- info | warn | critical
    headline       VARCHAR NOT NULL,
    detail         VARCHAR,
    created_at     TIMESTAMP DEFAULT current_timestamp
);

-- ------------------------------------------------------------ observability
CREATE TABLE IF NOT EXISTS traces (
    span_id        VARCHAR PRIMARY KEY,
    trace_id       VARCHAR NOT NULL,
    parent_id      VARCHAR,
    name           VARCHAR NOT NULL,
    kind           VARCHAR,               -- llm | tool | pipeline | agent | http
    status         VARCHAR,               -- ok | error
    started_at     TIMESTAMP,
    duration_ms    DOUBLE,
    model          VARCHAR,
    provider       VARCHAR,
    tokens_in      INTEGER,
    tokens_out     INTEGER,
    cost_usd       DOUBLE,
    cached         BOOLEAN,
    attrs          JSON,
    error          VARCHAR
);

-- --------------------------------------------------------------- evaluation
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id         VARCHAR PRIMARY KEY,
    suite          VARCHAR NOT NULL,
    task           VARCHAR NOT NULL,
    model          VARCHAR NOT NULL,
    provider       VARCHAR NOT NULL,
    prompt_name    VARCHAR,
    prompt_version VARCHAR,
    prompt_hash    VARCHAR,
    n_cases        INTEGER,
    metrics        JSON NOT NULL,
    git_sha        VARCHAR,
    started_at     TIMESTAMP,
    duration_s     DOUBLE,
    total_cost_usd DOUBLE,
    notes          VARCHAR
);

CREATE TABLE IF NOT EXISTS eval_results (
    result_id      VARCHAR PRIMARY KEY,
    run_id         VARCHAR NOT NULL,
    case_id        VARCHAR NOT NULL,
    expected       JSON,
    actual         JSON,
    correct        BOOLEAN,
    scores         JSON,
    latency_ms     DOUBLE,
    cost_usd       DOUBLE,
    error          VARCHAR
);

-- Human-in-the-loop review queue: cases the automated metrics flag as
-- low-confidence or ungrounded get surfaced here for a person to adjudicate,
-- and accepted verdicts flow back into the ground-truth set.
CREATE TABLE IF NOT EXISTS review_queue (
    review_id      VARCHAR PRIMARY KEY,
    extraction_id  VARCHAR NOT NULL,
    task           VARCHAR NOT NULL,
    reason         VARCHAR NOT NULL,      -- low_confidence | ungrounded | invalid | disagreement
    priority       INTEGER DEFAULT 0,
    status         VARCHAR DEFAULT 'open',-- open | accepted | corrected | rejected
    proposed       JSON,
    verdict        JSON,
    reviewer       VARCHAR,
    created_at     TIMESTAMP DEFAULT current_timestamp,
    resolved_at    TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_filings_cik ON filings(cik, filing_date);
CREATE INDEX IF NOT EXISTS idx_chunks_accession ON chunks(accession);
CREATE INDEX IF NOT EXISTS idx_sections_accession ON sections(accession);
CREATE INDEX IF NOT EXISTS idx_extractions_task ON extractions(task, cik);
CREATE INDEX IF NOT EXISTS idx_signals_lookup ON signals(name, cik, as_of);
CREATE INDEX IF NOT EXISTS idx_traces_trace ON traces(trace_id);
CREATE INDEX IF NOT EXISTS idx_eval_results_run ON eval_results(run_id);
