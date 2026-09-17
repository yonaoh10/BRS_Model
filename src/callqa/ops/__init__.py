"""Operational discipline that WRAPS the pipeline — never changes it.

Everything in this package reads the artifacts the core already writes and
writes new sidecar records beside them (the same pattern dashboard/ uses). It
imports the core; the core never imports it. Nothing here imports dashboard/ or
cloud/, so the one-command deletability of those add-ons is preserved.
"""
