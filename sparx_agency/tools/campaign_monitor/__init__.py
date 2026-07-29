"""Watching a long NavDP run — data collection and fine-tuning — from outside it.

Every reader here works off files the running job already writes: recording
directories and their ``meta.json``, the trainer's ``run.json`` and
``metrics.jsonl``. Nothing imports torch, nothing attaches to a process, and
nothing needs the job to cooperate, so the dashboard can be started, stopped and
restarted at any point in a multi-day campaign without disturbing it.
"""
