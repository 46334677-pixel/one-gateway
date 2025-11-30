# Cloudflare Tunnel（快速暴露 HTTPS）

适用于：没有域名/不方便开 443 端口，先把 `http://127.0.0.1:8000` 暴露为公网 HTTPS。

1) 安装 cloudflared（Debian/Ubuntu）
```bash
curl -fsSL https://pkg.cloudflare.com/install.sh | sudo bash
sudo apt-get install -y cloudflared
```

2) 启动本地开发服务
```bash
cd one-gateway
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

3) 启动 Quick Tunnel（生成临时 `https://xxxx.trycloudflare.com`）
```bash
cloudflared tunnel --url http://localhost:8000
```

4) 复制终端里显示的 `https://...trycloudflare.com` 地址，
   在 ChatGPT Actions 里选择 “Paste JSON/YAML” 更稳；
   或者在 “Import from URL” 里填 `https://.../openapi.json`。

提示：Quick Tunnel 域名会变化，用于临时调试非常方便；
上线后建议绑定你自己的域名并配置常驻隧道。

