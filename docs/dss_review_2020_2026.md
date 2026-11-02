# Review of *Decision Support Systems*, 2020-2026

The introduction of the article states that a review of *Decision Support Systems* (DSS) for 2020-2026 found no study
that optimises delivery promises against ratings. This file documents how that review was carried out and what it
found. The full positioning of the article against the closest studies inside and outside DSS is given in its
Table 1.

## Scope

- **Journal:** *Decision Support Systems*, ISSN 0167-9236.
- **Period:** articles published from 1 January 2020.
- **Dates:** main search 29 October 2026; update on 1 November 2026 covering the issues from June 2025 to volume 212
  (January 2027, articles in press).
- **Question:** does any DSS article (i) give distribution-free or quantile upper bounds on delivery time, (ii) treat
  the censoring of undelivered orders when such bounds are calibrated, or (iii) set delivery promises with an
  objective based on customer ratings?

## Search

1. **Crossref queries on the journal** (`api.crossref.org/journals/0167-9236/works`, `from-pub-date:2020-01-01`),
   in 20 query families:
   - delivery; delivery time, logistics, fulfilment
   - online review, rating; service failure, satisfaction
   - prediction interval, uncertainty; probabilistic forecasting, quantile; conformal prediction
   - survival, censored, time-to-event; hazard, duration
   - causal, treatment effect, uplift
   - prescriptive, decision-focused
   - concept drift, distribution shift
   - e-commerce, logistics, online retail
   - negative reviews, expectation; expectation, disconfirmation
   - ETA, travel time; delay, on-time
   - supply chain, lead time
   - marketplace, third-party; guarantee, commitment
   - cost-sensitive, newsvendor; calibration, reliability
   - policy learning, counterfactual; natural experiment, discontinuity
   - express, courier, logistics service
2. **Web searches restricted to DSS** for "conformal", "quantile regression", "prediction intervals" and "Olist".
3. **Update of 1 November 2026** (issues from June 2025): Crossref queries "delivery promise lead time logistics",
   "online review rating e-commerce customer" and "conformal prediction uncertainty censoring survival quantile".

## Screening

DSS articles were screened on their title and metadata and, where available, their abstract. An article was
recorded as related when it bears on one of the three research questions of the article: calibrated bounds on
delivery time under censoring and drift (RQ1), the response of ratings to the deviation from the promise (RQ2), or the
choice of promises (RQ3). References outside DSS that the article cites (methods, competing studies and evidence on
the value of delivery speed) were selected separately and are not covered by this review.

## Verification

- The DSS articles named in this file were checked against their Crossref records: title, all authors, year,
  volume and article number.
- Crossref does not hold the abstracts of Elsevier articles. Abstracts were read through Semantic Scholar or
  institutional repositories where available; the other DSS articles were characterised from their title and
  metadata.

## Findings

Within the scope above, no DSS article from 2020-2026 gives conformal or quantile upper bounds on delivery time, and
none optimises delivery promises against ratings. The closest DSS articles each address one part of the problem:

| Article | What it does | What it does not do |
|---|---|---|
| Ahmed, Deokar, Lee & Summerfield (2024), DSS 176, 114061 | Shows that orders sold with an express delivery promise are reviewed less often and rated lower than equally fast orders without one | No forecast of delivery time, no choice of promises |
| Yang, Zhang, Chen & Li (2023), DSS 175, 114042 | Costs and benefits of a delivery guarantee (characterised from title and metadata) | No forecast of delivery time |
| Kandula, Krishnamoorthy & Roy (2021), DSS 147, 113584 | Predicts delivery success and feeds the predictions to a scheduling model for e-commerce deliveries | No customer promise, no ratings |
| Wang, Xu, Feng, Ignatius, Yin & Xiao (2024), DSS 185, 114303 | Uplift model of the effect of appointment delivery on delivery success | No calibrated bounds on delivery dates |

The update of 1 November 2026 found articles on online food delivery and last-mile delivery published in 2025-2026,
none of which sets delivery promises or bounds delivery time.

## Limits

- Crossref indexes titles and metadata, not abstracts or full texts, and no full-text search of ScienceDirect was
  run; an article that addresses the question only in its body could have been missed.
- The findings describe the journal as of the dates above.
