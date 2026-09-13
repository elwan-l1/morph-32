# Frozen evaluation inputs

`wikitext2-valid.txt` supplies calibration only. `wikitext2-test.txt` supplies next-token scoring. Preserve bytes and tokenizer revision when comparing results.

- Validation SHA-256: `f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8`
- Test SHA-256: `d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0`

`mmlu-50.json` preserves complete five-shot prompts, gold labels, IDs, the selection seed and prompt hashes. Machine-specific source paths and selection bookkeeping are omitted. The whole-file subset hashes in archived measurement reports refer to the original JSON serialization, not this curated file. Use IDs and prompt hashes to verify evaluation identity.

The evaluation method is described in [the paper](../paper/morph32.pdf). Upstream datasets retain their original terms.
