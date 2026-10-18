"""The research service: 24/7 workers that read news, filings, market structure and
social chatter, turn what they find into structured notes, and hand the advisors a
per-symbol research view. Nothing in this package can place an order.

Layout
  db         the research database (SQLite WAL), shared by both boards and the service
  types      Item (what an adapter fetched) and Note (one labelled claim about one symbol)
  labels     the label set, the rule labeller and seed examples
  http       the polite HTTP client every adapter and the forever-search worker use
  text       RSS/Atom/HTML parsing, text extraction, near-duplicate fingerprints
  aliases    symbol mentions in text (BTC, "Bitcoin", $SOL, "(NASDAQ: NVDA)")
  classifier the extractor: a hashed linear text model trained in numpy
  sources/   source adapters (one class per source), grouped by team
  workers    worker slots, standbys, failover, gap detection and auto-growth
  aggregate  corroboration, penalties and the per-symbol research view
  outcome    outcome logging at 1h..7d horizons, source reliability, the outcome model
  rdesk      the advisors' learned research weights (engine side)
  stages     shadow -> active switches, one at a time, gated on evidence
  pretrain   replaying history so the advisors start trained
  cadence    the 2-hour cadence paper book that runs beside the main book
  thesis     display-only written theses (template, optional local model)
  forever    the sandboxed forever-search worker process and its supervisors
  service    the research process: lease, threads, timers, health
  link       the engine's narrow, fail-safe handle on the research database
"""
