import base64
import json


async def extract_image_base64_from_src(src: str, page) -> str:
    if src.startswith("data:image"):
        return src.split(",", 1)[1]
    else:
        resp = await page.evaluate(
            f"""async () => {{
                const r = await fetch({json.dumps(src)});
                const buf = await r.arrayBuffer();
                const bytes = new Uint8Array(buf);
                let binary = '';
                for (let i = 0; i < bytes.byteLength; i++) {{
                    binary += String.fromCharCode(bytes[i]);
                }}
                return btoa(binary);
            }}"""
        )
        img_bytes = base64.b64decode(resp)

    return base64.b64encode(img_bytes).decode()
