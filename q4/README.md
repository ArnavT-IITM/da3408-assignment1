# Question 4: Partner Reproduction

- Partner A: Aakash (DA24B028)
- Partner B: Arnav (DA24B027)
- Shared repository: [AakashAadhithya/aiops-assignment1](https://github.com/AakashAadhithya/aiops-assignment1)
- Arnav's reproduction commit: `95609cc`

I recreated Aakash's Conda environment and reran the experiment with `hidden_size=128`, `lr=0.001`, `batch_size=64`, `epochs=20`, and `seed=42`.

| Run | Validation accuracy | Training loss |
|---|---:|---:|
| Aakash's original run | 0.948 | 0.02032236113371596 |
| My reproduced run | 0.948 | 0.02032236113371596 |

The absolute difference in validation accuracy was `0.000`. This was within the stated tolerance of `0.001`, so the reproduction was a match.

Evidence:

- `evidence/partner_a_original_run.png`
- `evidence/partner_b_mlflow_run.png`
- The complete Partner B procedure and additional screenshots are in the shared repository under `partner_b/`.
