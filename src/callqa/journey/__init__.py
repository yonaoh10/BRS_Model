"""Customer journeys: repeat-contact analysis over a batch of interactions.

One account's contacts with the bank - recorded calls (a call may span several
audio files), written correspondence, calls that were never recorded - are
put on one timeline, beside what the bankers did on the account (Atlas), and
analysed as a story: why the customer came back, whether a promise was kept,
how many hands the case passed through, and how it ended.

Layout:
    models      the normalised dataset every importer produces
    importers   the generic CSV contract, the vendor-format workbook, Atlas
    nmf / assemble   NICE recordings, and joining a call's files into one call
    timeline / rules / metrics / analysis   the deterministic engine
    llm         interaction cards, return judgements, story narratives
"""
