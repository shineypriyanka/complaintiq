# Manual validation of the LLM judge (Phase 9)

Reviewed 12 claims: both claims the judge flagged as partial, plus 10 random "supported" claims.

| # | Judge | Human | Note |
|---|---|---|---|
| 1 | partial | partial | Dropped the RBI/NPCI/Card Network timeline condition |
| 2 | partial | exclude | Bank's own promise, not a regulatory claim (judge scoping error) |
| 3 | yes | yes | |
| 4 | yes | yes | Judge wrongly listed [2] as supporting |
| 5 | yes | yes | Same claim as #4; judge correctly excluded [2] (inconsistent) |
| 6 | yes | yes | |
| 7 | yes | yes | |
| 8 | yes | partial | Omits "to the Regulated Entity": rule applied to the wrong complaint |
| 9 | yes | yes | |
| 10 | yes | partial | Escalation condition restated as a bank obligation |
| 11 | yes | yes | |
| 12 | yes | yes | |

**Agreement: 9/12 (75%).** The judge is lenient on qualifier-dropping errors
(consistent with self-preference bias: the judge is the same model as the generator),
so the 96.8% strict faithfulness should be read as an upper bound.