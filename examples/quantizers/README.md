# One label, nine different files

Three model families, same question: what is actually inside a
file that calls itself Q4_K_M. Each card here is current
`picchio id` output.

## Qwen3.5-9B: four quantizers

Same base model (Qwen/Qwen3.5-9B), same Q4_K_M label, four
published GGUFs.

| quantizer | file | bytes | sha256 (first 12) |
|---|---|---|---|
| unsloth | Qwen3.5-9B-Q4_K_M.gguf | 5,680,522,464 | 03b74727a860 |
| bartowski | Qwen_Qwen3.5-9B-Q4_K_M.gguf | 6,169,341,984 | d784ce9eda1a |
| lmstudio-community | Qwen3.5-9B-Q4_K_M.gguf | 5,627,044,256 | cd76ec205963 |
| mradermacher | Qwen3.5-9B.Q4_K_M.gguf | 5,627,045,120 | 9fa52e37c829 |

Over the 427 tensors all four files share, effective bits per
weight: lmstudio-community 5.02, mradermacher 5.02, unsloth 5.07,
bartowski 5.27. The bartowski card reads 5.36 whole-file because
that file bundles a 243M-parameter MTP head at q8_0; unsloth ships
the same head as a separate repo.

The unsloth file was walked in full and its sha256 matches the
repo's lfs oid. The other three were walked from the first 16 MiB
(the tensor table lives in the header) plus the file size; all
four sizes were verified byte-exact against the CDN's
Content-Range. Rerun any of it: `./picchio id <file>`.

Retrieved 2026-07-14, revision main. The unsloth sha256 above was
confirmed a second time on 2026-08-22, by a different machine
downloading the same file fresh from the same repo.

## Qwen3.5-4B: three published files

| publisher | file | bytes | sha256 (first 12) | effective bpw |
|---|---|---:|---|---:|
| lmstudio-community | Qwen3.5-4B-Q4_K_M.gguf | 2,707,513,696 | 25082a7dd377 | 5.13 |
| unsloth | Qwen3.5-4B-Q4_K_M.gguf | 2,740,937,888 | 00fe7986ff5f | 5.19 |
| bartowski | Qwen_Qwen3.5-4B-Q4_K_M.gguf | 3,013,027,808 | 13c16f426047 | 5.55 |

Cards: [lmstudio-community-4b.txt](lmstudio-community-4b.txt) ·
[unsloth-4b.txt](unsloth-4b.txt) ·
[bartowski-4b.txt](bartowski-4b.txt).

All three were downloaded in full from pinned upstream commits,
and each full SHA-256 matched the repository's LFS object id.
The spread inside one label is 8.2%. The bartowski file also
contains 120,599,552 more weights than the other two; its own
origin fields name a different base-model variant. The card
reports that difference instead of treating the filename as
identity.

## Qwen3.8-27B: two quantizers

| quantizer | file | bytes | sha256 (first 12) |
|---|---|---|---|
| unsloth (UD) | Qwen3.8-27B-UD-Q4_K_M.gguf | 16,464,440,224 | 322e194ff797 |
| lmstudio-community | Qwen3.8-27B-Q4_K_M.gguf | 16,810,714,336 | e00082f779fa |

Cards: [unsloth-27b.txt](unsloth-27b.txt) ·
[lmstudio-community-27b.txt](lmstudio-community-27b.txt).

Both files say Q4_K_M. The unsloth Dynamic build spends its bits
over nine tensor types and prices out at 4.82 bits per weight;
the lmstudio-community build uses three types and prices out at
4.92. Same label, 0.10 bit apart, 346 MB apart, and the heavier
file is the one with the plainer recipe.

They differ in a second way the cards make visible: the unsloth
file carries origin keys (`quantized_by`, `base_model`) and the
lmstudio-community file carries none, so its card says who made
it cannot be read off the file. That is a fact about the file,
not a complaint about the publisher.

Both were downloaded in full and both sha256 values were checked
against the repo's own lfs oid before the walk; `picchio id`
then recomputed each sha independently and got the same answer.
Retrieved 2026-08-22, revision pinned per file.
