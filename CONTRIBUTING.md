# Contributing

Thanks for helping improve this list. The goal is a small, accurate, well-annotated collection, not an exhaustive one.

## Scope

In scope:
- Reinforcement learning or reward design that targets calibration, confidence, or uncertainty.
- Abstention, selective prediction, and hallucination-aware training for language or decision models.
- Uncertainty-aware or calibrated reward models.
- Decision-making under uncertainty with language models and agents.
- Typed or non-generative decision models (for example, System One models such as Jev) and independent evaluations of them.
- Foundational calibration and scoring-rule work that the above builds on.

Out of scope: general RL without a calibration angle, pure computer-vision calibration (unless foundational), product marketing, and anything you are not free to share publicly.

## How to suggest a paper

Open an issue with the **Paper suggestion** template, or send a pull request.

## Entry format (README)

```
- [Title](canonical URL) (Venue Year · or *preprint*). One-sentence takeaway.
```

- Use a canonical URL: arXiv abstract page, ACL Anthology, OpenReview, PMLR, proceedings site, or DOI.
- Put the entry in the right part, newest first.
- The takeaway states what the paper shows, in neutral language. No hype.

## Also update

1. `papers.csv`: one row with title, authors, year, venue, venue_status, theme, method, key finding, limitations, URL.
2. `papers.bib`: one BibTeX entry.

## Verification rules

- The URL must resolve.
- A venue is listed only if it can be confirmed on a proceedings, ACL Anthology, PMLR, publisher, or accepted-paper page. Otherwise mark the entry *preprint*, or *under review / unconfirmed* if only a submission listing exists.
- Vendor claims go under "Vendor claims" and must not be presented as independent results.

## Pull request checklist

- [ ] Not a duplicate
- [ ] Correct part and ordering
- [ ] URL resolves; venue verified or marked
- [ ] `papers.csv` and `papers.bib` updated
- [ ] No promotional or affiliate links

## Conduct

Be respectful and assume good faith. This project follows the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/), version 2.1.
