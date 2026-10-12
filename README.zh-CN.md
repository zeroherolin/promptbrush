# promptbrush

[English](README.md) | **简体中文**

以提示词为笔：调用 OpenAI 兼容 `/v1/images` 接口的单文件命令行工具与 Python 库。支持根据提示词生成图片，或基于一张、多张参考图编辑。

想在浏览器里圈选局部、把每次修改留在版本树上，可以用基于本项目的 [brushgraft](https://github.com/zeroherolin/brushgraft)。

```bash
promptbrush '一只戴着宇航员头盔的柯基犬，皮克斯风格'
promptbrush -i cat.png -i hat.png '让这只猫戴上这顶帽子'
```

## 特性

- 用 `-i` 传入图片或目录即为编辑，否则为生成。
- `-n N` 拆成 N 个单张请求，最多 4 个并发，单张失败不影响其余。
- 网络错误、429、500–504 与 Cloudflare 52x 按指数退避重试，并按 `Retry-After` 等待（最多一分钟）。
- 尺寸、参考图和保存目录在本地先检查，有误时不发出付费请求。
- 日志写入 stderr，stdout 每行一个已保存的文件路径。
- `create_image` 返回图片字节，可在 Python 中调用，见[作为 Python 库使用](#作为-python-库使用)。
- 单文件，只依赖 `requests`，Python 3.10+。

## 安装

```bash
git clone https://github.com/zeroherolin/promptbrush.git
cd promptbrush
pip install -e .          # 或：pip install git+https://github.com/zeroherolin/promptbrush.git
cp config.example .env    # 然后填入 API Key
```

| 变量 | 说明 |
| --- | --- |
| `PROMPTBRUSH_API_KEY` | 必填 |
| `PROMPTBRUSH_BASE_URL` | 服务地址，填到 `/v1` 为止；默认 `https://api.openai.com/v1` |

环境变量优先于 `.env`；`.env` 从 `promptbrush.py` 所在目录读取，装到 site-packages 时请改用环境变量。

## 使用

```text
promptbrush [prompt ...] [-i PATH] [-s SIZE] [-n NUM] [-m MODEL] [-o DIR]
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `prompt` | | 提示词；省略时读取脚本目录下的 `prompt.md` |
| `-i, --image` | | 参考图片或目录，可重复；指定后进入编辑模式 |
| `-s, --size` | `1600x1200` | `宽x高` 或 `auto` |
| `-n, --num` | `1` | 生成数量，1 到 10 |
| `-m, --model` | `gpt-image-2.5-flare` | 模型名称 |
| `-o, --output` | `.` | 保存目录 |

```bash
promptbrush -s 3840x2160 -n 2 -o out/ '清晨的江南水乡，石桥倒影，薄雾'
promptbrush -i ./refs/ '参考这些图片的风格，画一座灯塔'
promptbrush '一只猫' > saved.txt    # stdout 每行一个已保存的路径
```

退出码：至少保存了一张为 `0`；一张也没保存，或尺寸、路径等取值无效为 `1`；参数格式错误为 `2`；Ctrl+C 中断为 `130`。设置 `NO_COLOR` 可关闭日志配色。

## 模型

2026 年 10 月在一个 OpenAI 兼容中转服务上的实测结果；可用模型取决于你的服务商。

| 模型 | 单张耗时 | 输出尺寸 | 编辑时与原图对齐 | 备注 |
| --- | --- | --- | --- | --- |
| `gpt-image-2.5-sunburst` | 40–90s | 精确，最大 4K | 是 | 画质最佳 |
| `gpt-image-2.5-flare` | 25–45s | 精确，最大 4K | 是 | **默认** |
| `gpt-image-2` | 20–40s | 只取宽高比，约 157 万像素 | 是 | 审核最宽松 |
| `gpt-image-2-vip` | 30–75s | 多数尺寸精确，最大 4K | 是 | 唯一遵守 `mask` 的模型 |
| `gpt-image-1.5` | 15–30s | 只取宽高比，约 157 万像素 | 是 | |
| `gemini-nano-banana-2.1` | 15–30s | 1024×1024 | 否 | |
| `gemini-3-pro-image` | 25–45s | 1024×1024 | 否 | 即 Nano Banana Pro |
| `gemini-3.1-flash-image` | 10–20s | 1024×1024 | 否 | 最快 |
| `jimeng-4.5` | 40–55s | 只取宽高比，约 415 万像素 | 是 | 带“AI”角标水印 |

“对齐”指编辑结果保持原图取景，可以逐像素对照或贴回原图。`gpt-image-2.5-*` 要求宽高都是 16 的倍数、单边不超过 3840、宽高比不超过 3:1、总像素在 655,360–8,294,400 之间，promptbrush 会在本地预先检查。这张表也以 `promptbrush.MODELS` 提供，并列在 `--help` 末尾。

## 作为 Python 库使用

```python
import promptbrush as pb

gen = pb.create_image("一只戴着宇航员头盔的柯基犬", size="1536x1024")

photo = open("cat.png", "rb").read()
gen = pb.create_image("给猫戴上一顶红色毛线帽", images=[photo],
                      size=pb.fit_size(1600, 1200))  # 与原图同比例
open("out.png", "wb").write(gen.images[0])
```

每次调用发送一个请求，重试已内置。输入无效时在发送前抛 `ValueError`；接口失败（包括响应里没有图片）抛 `pb.ImageAPIError`，带 `.status`、`.code` 与 `.moderated`；用 `with pb.log_sink(callback):` 可以接管日志。

## 注意事项

- 超时或网关 524 后重试的请求，每一次都可能计费：服务端可能已经完成了之前的请求。
- 内容审核拦截时，实测的中转服务返回 HTTP 451，OpenAI 返回 `400 moderation_blocked`；不同模型的通过率差别很大。

## 开发

```bash
python -m unittest -v test_promptbrush    # 全部离线，约 1 秒
```

## 许可证

MIT — 见 [LICENSE](LICENSE)。
