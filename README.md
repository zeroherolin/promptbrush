# promptbrush

**English** | [简体中文](README.zh-CN.md)

Paint with prompts: a single-file CLI and Python library for OpenAI-compatible `/v1/images` APIs. Generate images from a prompt, or edit one or more reference images.

To mark regions in a browser and keep every edit in a version tree, see [brushgraft](https://github.com/zeroherolin/brushgraft), which is built on promptbrush.

```bash
promptbrush 'a corgi wearing an astronaut helmet, Pixar style'
promptbrush -i cat.png -i hat.png 'put this hat on this cat'
```

## Features

- **Generate or edit**: pass images or a directory with `-i` to edit instead of generate.
- **Reliable batches**: `-n N` sends N single-image requests, 4 at a time, so one failure doesn't cost the rest.
- **Retries**: network errors, 429 and 5xx / Cloudflare 52x are retried with backoff, honoring `Retry-After`.
- **Checked before paying**: invalid sizes, options and reference images fail locally, before any request.
- **Script-friendly**: logs go to stderr; stdout lists only the saved files.
- **Also a library**: `create_image` returns image bytes; see [Python library](#python-library).
- One file, one dependency (`requests`), Python 3.10+.

## Installation

```bash
git clone https://github.com/zeroherolin/promptbrush.git
cd promptbrush
pip install -e .          # or: pip install git+https://github.com/zeroherolin/promptbrush.git
cp config.example .env    # then fill in your API key
```

| Variable | Description |
| --- | --- |
| `PROMPTBRUSH_API_KEY` | Required |
| `PROMPTBRUSH_BASE_URL` | Base URL up to `/v1`; default `https://api.openai.com/v1` |

Environment variables override `.env`, which is read from the directory of `promptbrush.py`. When installed into site-packages, use environment variables.

## Usage

```text
promptbrush [prompt ...] [-i PATH] [-s SIZE] [-n NUM] [-m MODEL] [-o DIR]
```

| Option | Default | Description |
| --- | --- | --- |
| `prompt` | `prompt.md` | Prompt text; read from `prompt.md` next to the script when omitted |
| `-i, --image` | | Reference image or directory; repeatable; switches to edit mode |
| `-s, --size` | `1600x1200` | `WIDTHxHEIGHT` or `auto` |
| `-n, --num` | `1` | Number of images |
| `-m, --model` | `gpt-image-2.5-flare` | Model name |
| `-o, --output` | `.` | Output directory |

```bash
promptbrush -s 3840x2160 -n 2 -o out/ 'a misty canal town at dawn'
promptbrush -i ./refs/ 'a lighthouse in the style of these images'
promptbrush 'a cat' > saved.txt    # stdout is one saved path per line
```

Exit codes: `0` success (at least one image saved), `1` failure, `2` invalid arguments, `130` Ctrl+C. Set `NO_COLOR` to turn off colored logs.

## Models

Measured in October 2026 on one OpenAI-compatible relay; what's available depends on your provider.

| Model | Time | Output size | Aligned in edits | Notes |
| --- | --- | --- | --- | --- |
| `gpt-image-2.5-sunburst` | 40–90s | Exact, ≤4K | Yes | Best quality |
| `gpt-image-2.5-flare` | 25–45s | Exact, ≤4K | Yes | **Default** |
| `gpt-image-2` | 20–40s | Aspect only, ~1.57 MP | Yes | Most lenient moderation |
| `gpt-image-2-vip` | 30–75s | Exact for most sizes, ≤4K | Yes | Only model that respects `mask` |
| `gpt-image-1.5` | 15–30s | Aspect only, ~1.57 MP | Yes | |
| `gemini-nano-banana-2.1` | 15–30s | 1024×1024 | No | |
| `gemini-3-pro-image` | 25–45s | 1024×1024 | No | Nano Banana Pro |
| `gemini-3.1-flash-image` | 10–20s | 1024×1024 | No | Fastest |
| `jimeng-4.5` | 40–55s | Aspect only, ~4.15 MP | Yes | Adds an "AI" watermark |

"Aligned" means an edit keeps the original framing, so the result can be compared or pasted back pixel by pixel. `gpt-image-2.5-*` need width and height in multiples of 16, sides ≤ 3840, a ratio ≤ 3:1 and 655,360–8,294,400 pixels; promptbrush checks this locally. The table is also available as `promptbrush.MODELS` and at the end of `--help`.

## Python library

```python
import promptbrush as pb

gen = pb.create_image("a corgi wearing an astronaut helmet", size="1536x1024")

photo = open("cat.png", "rb").read()
gen = pb.create_image("give the cat a red knitted hat", images=[photo],
                      size=pb.fit_size(1600, 1200))  # same aspect as the photo
open("out.png", "wb").write(gen.images[0])
```

Each call sends one request, with retries built in. Invalid input raises `ValueError` before anything is sent, API failures raise `pb.ImageAPIError` (with `.status` and `.moderated`), and `with pb.log_sink(callback):` captures the log lines.

## Notes

- A retry after a timeout or a 524 may be billed twice: the server may already have generated the first image.
- Moderation blocks answer with HTTP 451, and pass rates differ a lot between models.

## Development

```bash
python -m unittest -v test_promptbrush    # offline, about a second
```

## License

MIT — see [LICENSE](LICENSE).
